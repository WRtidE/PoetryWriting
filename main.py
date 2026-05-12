#!/usr/bin/env python3
"""
PoetryWriting - 自动古诗生成 GUI
基于 PyQt5 的图形界面，支持诗句续写与藏头诗生成。
"""

import sys
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QSlider, QGroupBox, QComboBox,
    QMessageBox, QProgressBar, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QTextCursor, QFontDatabase


from model import get_model


# ========== 配置 ==========

class Config:
    max_gen_len = 64


# 各模型默认超参数
# 注：train.py 保存模型时会在文件名后追加损失率（如 poetry_model_GRU_1.3569.pth），
#     请将最佳模型重命名为以下路径，或修改此处指向实际文件。
MODEL_CONFIGS = {
    'lstm': {'embedding_dim': 256, 'hidden_dim': 512, 'model_path': 'result/poetry_model_LSTM.pth'},
    'gru':  {'embedding_dim': 256, 'hidden_dim': 512, 'model_path': 'result/poetry_model_GRU.pth'},
}


# ========== 模型加载与生成线程 ==========

class PoetryWorker(QThread):
    """后台线程：加载模型 / 生成诗句，避免阻塞 UI"""

    # 信号
    model_loaded = pyqtSignal(object, object, object)  # model, ix2word, word2ix
    poem_generated = pyqtSignal(str)                     # 生成的诗句
    error_occurred = pyqtSignal(str)                     # 错误信息
    progress_update = pyqtSignal(str)                    # 状态更新

    def __init__(self):
        super().__init__()
        self.mode = None          # 'load' | 'generate' | 'acrostic'
        self.model_type = 'gru'   # 当前模型类型
        self.start_words = ''
        self.head_words = ''
        self.temperature = 1.0
        self.top_k = 50

    # ---- 后台逻辑 ----

    def load_model_and_data(self):
        """加载字典和模型"""
        self.progress_update.emit('正在加载唐诗数据集...')

        datas = np.load('dataset/tang.npz', allow_pickle=True)
        ix2word = datas['ix2word'].item()
        word2ix = datas['word2ix'].item()

        self.progress_update.emit(f'正在加载 {self.model_type.upper()} 预训练模型...')

        cfg = MODEL_CONFIGS[self.model_type]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(
            model_type=self.model_type,
            vocab_size=len(word2ix),
            embedding_dim=cfg['embedding_dim'],
            hidden_dim=cfg['hidden_dim'],
        ).to(device)
        model.load_state_dict(
            torch.load(cfg['model_path'], map_location=device, weights_only=True)
        )
        model.eval()

        self.model_loaded.emit(model, ix2word, word2ix)

    def generate_poem(self, model, ix2word, word2ix):
        """续写诗句（温度 + top-k 采样）"""
        device = next(model.parameters()).device
        results = list(self.start_words)
        start_len = len(self.start_words)

        input_tensor = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)
        hidden = None

        with torch.no_grad():
            for i in range(Config.max_gen_len):
                output, hidden = model(input_tensor, hidden)

                if i < start_len:
                    word = results[i]
                    input_tensor = torch.Tensor([word2ix[word]]).view(1, 1).long().to(device)
                else:
                    logits = output.data[0] / self.temperature
                    top_k_logits, top_k_indices = logits.topk(self.top_k)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    top_index = top_k_indices[torch.multinomial(probs, 1).item()].item()
                    word = ix2word[top_index]

                    if word == '<EOP>' and len(results) > 20:
                        break
                    results.append(word)
                    input_tensor = torch.Tensor([top_index]).view(1, 1).long().to(device)

        self.poem_generated.emit(''.join(results))

    def generate_acrostic(self, model, ix2word, word2ix):
        """生成藏头诗"""
        device = next(model.parameters()).device
        results = []
        hidden = None
        head_index = 0
        sentence_len = 0
        head_words = self.head_words

        input_tensor = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)

        with torch.no_grad():
            while True:
                output, hidden = model(input_tensor, hidden)

                if sentence_len == 0:
                    if head_index >= len(head_words):
                        break
                    word = head_words[head_index]
                    head_index += 1
                else:
                    p = torch.softmax(output / self.temperature, dim=1)
                    top_index = torch.multinomial(p, 1).item()
                    word = ix2word[top_index]
                    if word in ['<EOP>', '<START>', '</s>', '，', '。']:
                        continue

                results.append(word)
                sentence_len += 1

                if sentence_len == 7:
                    results.append('，' if head_index % 2 == 1 else '。')
                    sentence_len = 0

                input_tensor = torch.Tensor([word2ix[word]]).view(1, 1).long().to(device)

        self.poem_generated.emit(''.join(results))

    # ---- 线程入口 ----

    def run(self):
        try:
            if self.mode == 'load':
                self.load_model_and_data()
            elif self.mode == 'generate':
                self.progress_update.emit('正在构思诗句...')
                self.generate_poem(self.model, self.ix2word, self.word2ix)
            elif self.mode == 'acrostic':
                self.progress_update.emit('正在构思藏头诗...')
                self.generate_acrostic(self.model, self.ix2word, self.word2ix)
        except Exception as e:
            self.error_occurred.emit(str(e))


# ========== 主窗口 ==========

class PoetryWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.ix2word = None
        self.word2ix = None
        self.worker = None
        self.model_type = 'gru'   # 默认模型
        self._init_ui()
        self._apply_style()
        # 启动后自动加载模型
        QTimer.singleShot(100, self._load_model)

    # ---- UI 布局 ----

    def _init_ui(self):
        self.setWindowTitle('📜 自动古诗生成')
        
        # 允许窗口自适应缩放
        self.resize(680, 760)
        self.setMinimumSize(600, 700) 

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        # 标题
        title = QLabel('自动古诗生成')
        title.setObjectName('titleLabel')
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel(f'基于 {self.model_type.upper()} 的唐诗生成模型 · 续写 & 藏头诗')
        subtitle.setObjectName('subtitleLabel')
        subtitle.setAlignment(Qt.AlignCenter)
        self.subtitle_label = subtitle
        root.addWidget(subtitle)

        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName('divider')
        root.addWidget(line)

        # 模型选择栏
        model_row = QHBoxLayout()
        model_row.setSpacing(10)
        model_label = QLabel('🤖 模型选择')
        model_label.setObjectName('modelLabel')
        model_row.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.setObjectName('modelCombo')
        self.model_combo.addItem('GRU（推荐）', 'gru')
        self.model_combo.addItem('LSTM', 'lstm')
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_row.addWidget(self.model_combo)

        self.load_btn = QPushButton('🔄 重新加载模型')
        self.load_btn.setObjectName('loadBtn')
        self.load_btn.clicked.connect(self._load_model)
        self.load_btn.setEnabled(False)
        model_row.addWidget(self.load_btn)
        model_row.addStretch()
        root.addLayout(model_row)

        # 状态栏
        self.status_label = QLabel('正在初始化...')
        self.status_label.setObjectName('statusLabel')
        self.status_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.status_label)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)          # 无限滚动
        self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # 标签页
        self.tabs = QTabWidget()
        self.tabs.setObjectName('mainTabs')
        self.tabs.addTab(self._build_continuation_tab(), '  续写诗句  ')
        self.tabs.addTab(self._build_acrostic_tab(), '  藏 头 诗  ')
        root.addWidget(self.tabs)

    # ---- 续写诗句 Tab ----

    def _build_continuation_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # 输入区
        input_group = QGroupBox('📝 输入诗句开头')
        input_group.setObjectName('groupBox')
        ig = QVBoxLayout(input_group)
        ig.setSpacing(10)

        self.cont_input = QLineEdit()
        self.cont_input.setPlaceholderText('例：床前明月光 / 春风得意马蹄疾')
        self.cont_input.setObjectName('inputField')
        self.cont_input.returnPressed.connect(self._on_generate_continuation)
        ig.addWidget(self.cont_input)

        layout.addWidget(input_group)

        # 参数区
        param_group = QGroupBox('🎛 生成参数')
        param_group.setObjectName('groupBox')
        pg = QVBoxLayout(param_group)
        pg.setSpacing(10)

        # 温度
        t_row = QHBoxLayout()
        t_row.addWidget(QLabel('温度 (Temperature)'))
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(2, 20)         # 0.2 ~ 2.0
        self.temp_slider.setValue(10)
        self.temp_slider.setTickPosition(QSlider.TicksBelow)
        self.temp_slider.setTickInterval(2)
        self.temp_value = QLabel('1.0')
        self.temp_value.setFixedWidth(40)
        self.temp_value.setAlignment(Qt.AlignCenter)
        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_value.setText(f'{v / 10:.1f}')
        )
        t_row.addWidget(self.temp_slider)
        t_row.addWidget(self.temp_value)
        pg.addLayout(t_row)

        # Top-K
        k_row = QHBoxLayout()
        k_row.addWidget(QLabel('Top-K'))
        self.topk_slider = QSlider(Qt.Horizontal)
        self.topk_slider.setRange(5, 200)
        self.topk_slider.setValue(50)
        self.topk_slider.setTickPosition(QSlider.TicksBelow)
        self.topk_slider.setTickInterval(20)
        self.topk_value = QLabel('50')
        self.topk_value.setFixedWidth(40)
        self.topk_value.setAlignment(Qt.AlignCenter)
        self.topk_slider.valueChanged.connect(
            lambda v: self.topk_value.setText(str(v))
        )
        k_row.addWidget(self.topk_slider)
        k_row.addWidget(self.topk_value)
        pg.addLayout(k_row)

        layout.addWidget(param_group)

        # 生成按钮
        self.cont_btn = QPushButton('✨ 生成诗句')
        self.cont_btn.setObjectName('generateBtn')
        self.cont_btn.clicked.connect(self._on_generate_continuation)
        self.cont_btn.setEnabled(False)
        layout.addWidget(self.cont_btn)

        # 输出区
        out_group = QGroupBox('📖 生成结果')
        out_group.setObjectName('groupBox')
        og = QVBoxLayout(out_group)
        self.cont_output = QTextEdit()
        self.cont_output.setReadOnly(True)
        self.cont_output.setPlaceholderText('等待生成...')
        self.cont_output.setObjectName('outputArea')
        
        # 允许纵向拉伸
        self.cont_output.setMinimumHeight(150)
        
        og.addWidget(self.cont_output)

        # 复制按钮
        copy_row = QHBoxLayout()
        copy_row.addStretch()
        copy_btn = QPushButton('📋 复制')
        copy_btn.setObjectName('copyBtn')
        copy_btn.clicked.connect(lambda: self._copy_to_clipboard(self.cont_output))
        copy_row.addWidget(copy_btn)
        og.addLayout(copy_row)

        layout.addWidget(out_group)
        return w

    # ---- 藏头诗 Tab ----

    def _build_acrostic_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # 输入区
        input_group = QGroupBox('📝 输入藏头字')
        input_group.setObjectName('groupBox')
        ig = QVBoxLayout(input_group)
        ig.setSpacing(10)

        self.acro_input = QLineEdit()
        self.acro_input.setPlaceholderText('例：我爱编程（支持 2-8 字）')
        self.acro_input.setObjectName('inputField')
        self.acro_input.returnPressed.connect(self._on_generate_acrostic)
        ig.addWidget(self.acro_input)

        layout.addWidget(input_group)

        # 参数区
        param_group = QGroupBox('🎛 生成参数')
        param_group.setObjectName('groupBox')
        pg = QVBoxLayout(param_group)
        pg.setSpacing(10)

        t_row = QHBoxLayout()
        t_row.addWidget(QLabel('温度 (Temperature)'))
        self.acro_temp_slider = QSlider(Qt.Horizontal)
        self.acro_temp_slider.setRange(2, 20)
        self.acro_temp_slider.setValue(8)
        self.acro_temp_slider.setTickPosition(QSlider.TicksBelow)
        self.acro_temp_slider.setTickInterval(2)
        self.acro_temp_value = QLabel('0.8')
        self.acro_temp_value.setFixedWidth(40)
        self.acro_temp_value.setAlignment(Qt.AlignCenter)
        self.acro_temp_slider.valueChanged.connect(
            lambda v: self.acro_temp_value.setText(f'{v / 10:.1f}')
        )
        t_row.addWidget(self.acro_temp_slider)
        t_row.addWidget(self.acro_temp_value)
        pg.addLayout(t_row)

        layout.addWidget(param_group)

        # 生成按钮
        self.acro_btn = QPushButton('✨ 生成藏头诗')
        self.acro_btn.setObjectName('generateBtn')
        self.acro_btn.clicked.connect(self._on_generate_acrostic)
        self.acro_btn.setEnabled(False)
        layout.addWidget(self.acro_btn)

        # 输出区
        out_group = QGroupBox('📖 生成结果')
        out_group.setObjectName('groupBox')
        og = QVBoxLayout(out_group)
        self.acro_output = QTextEdit()
        self.acro_output.setReadOnly(True)
        self.acro_output.setPlaceholderText('等待生成...')
        self.acro_output.setObjectName('outputArea')
        
        # 允许纵向拉伸
        self.acro_output.setMinimumHeight(150)
        
        og.addWidget(self.acro_output)

        copy_row = QHBoxLayout()
        copy_row.addStretch()
        copy_btn = QPushButton('📋 复制')
        copy_btn.setObjectName('copyBtn')
        copy_btn.clicked.connect(lambda: self._copy_to_clipboard(self.acro_output))
        copy_row.addWidget(copy_btn)
        og.addLayout(copy_row)

        layout.addWidget(out_group)
        return w

    # ---- 模型加载 ----

    def _on_model_changed(self, index):
        """模型下拉框切换时更新 model_type，但不自动加载"""
        self.model_type = self.model_combo.currentData()
        self.subtitle_label.setText(
            f'基于 {self.model_type.upper()} 的唐诗生成模型 · 续写 & 藏头诗'
        )

    def _load_model(self):
        """读取当前选择的模型类型并加载"""
        self.model_type = self.model_combo.currentData()
        self.subtitle_label.setText(
            f'基于 {self.model_type.upper()} 的唐诗生成模型 · 续写 & 藏头诗'
        )

        self.status_label.setText('正在加载模型...')
        self.progress.setVisible(True)
        self.load_btn.setEnabled(False)
        self.cont_btn.setEnabled(False)
        self.acro_btn.setEnabled(False)

        self.worker = PoetryWorker()
        self.worker.model_type = self.model_type
        self.worker.mode = 'load'
        self.worker.model_loaded.connect(self._on_model_loaded)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.progress_update.connect(self.status_label.setText)
        self.worker.start()

    def _on_model_loaded(self, model, ix2word, word2ix):
        self.model = model
        self.ix2word = ix2word
        self.word2ix = word2ix
        self.progress.setVisible(False)
        self.status_label.setText(f'✅ {self.model_type.upper()} 模型就绪 — 可以开始生成')
        self.load_btn.setEnabled(True)
        self.cont_btn.setEnabled(True)
        self.acro_btn.setEnabled(True)

    # ---- 生成回调 ----

    def _on_generate_continuation(self):
        text = self.cont_input.text().strip()
        if not text:
            QMessageBox.warning(self, '提示', '请输入诗句开头。')
            return
        # 检查词汇
        unknown = [ch for ch in text if ch not in self.word2ix]
        if unknown:
            QMessageBox.warning(
                self, '提示',
                f'以下字符不在词表中：{" ".join(unknown)}\n请更换输入。'
            )
            return
        self._start_generation('generate', start_words=text)

    def _on_generate_acrostic(self):
        text = self.acro_input.text().strip()
        if not text:
            QMessageBox.warning(self, '提示', '请输入藏头字。')
            return
        if len(text) < 2 or len(text) > 8:
            QMessageBox.warning(self, '提示', '藏头字建议 2-8 个。')
            return
        unknown = [ch for ch in text if ch not in self.word2ix]
        if unknown:
            QMessageBox.warning(
                self, '提示',
                f'以下字符不在词表中：{" ".join(unknown)}\n请更换输入。'
            )
            return
        self._start_generation('acrostic', head_words=text)

    def _start_generation(self, mode, start_words='', head_words=''):
        # 禁用按钮，显示进度
        self.cont_btn.setEnabled(False)
        self.acro_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status_label.setText('正在生成...')

        self.worker = PoetryWorker()
        self.worker.mode = mode
        self.worker.model = self.model
        self.worker.ix2word = self.ix2word
        self.worker.word2ix = self.word2ix
        self.worker.start_words = start_words
        self.worker.head_words = head_words
        if mode == 'generate':
            self.worker.temperature = self.temp_slider.value() / 10.0
            self.worker.top_k = self.topk_slider.value()
        else:
            self.worker.temperature = self.acro_temp_slider.value() / 10.0
        self.worker.poem_generated.connect(self._on_poem_generated)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

    def _on_poem_generated(self, poem):
        self.progress.setVisible(False)
        self.status_label.setText('✅ 生成完成')
        self.cont_btn.setEnabled(True)
        self.acro_btn.setEnabled(True)

        # 判断当前 tab
        current = self.tabs.currentIndex()
        if current == 0:
            output = self.cont_output
        else:
            output = self.acro_output

        output.clear()
        # 设置大号字体显示诗句
        output.setFontPointSize(18)
        output.setTextColor(QColor('#2c1810'))
        output.append(poem)
        # 滚动到顶部
        output.moveCursor(QTextCursor.Start)

    def _on_error(self, msg):
        self.progress.setVisible(False)
        self.status_label.setText('❌ 发生错误')
        self.cont_btn.setEnabled(True)
        self.acro_btn.setEnabled(True)
        QMessageBox.critical(self, '错误', f'生成失败：\n{msg}')

    def _copy_to_clipboard(self, widget):
        text = widget.toPlainText().strip()
        if text:
            QApplication.clipboard().setText(text)
            self.status_label.setText('📋 已复制到剪贴板')
            QTimer.singleShot(2000, lambda: self.status_label.setText('✅ 模型就绪 — 可以开始生成'))

    # ---- 样式 ----

    def _apply_style(self):
        self.setStyleSheet("""
            /* ===== 全局 ===== */
            QWidget {
                background-color: #fdf6ec;
                font-family: "STSong", "Songti SC", "SimSun", "Noto Serif CJK SC", serif;
            }

            /* ===== 标题 ===== */
            #titleLabel {
                font-size: 30px;
                font-weight: bold;
                color: #5c3d2e;
                padding: 8px 0;
                letter-spacing: 8px;
            }

            #subtitleLabel {
                font-size: 13px;
                color: #a08060;
                padding-bottom: 4px;
            }

            #divider {
                background-color: #d4b896;
                max-height: 1px;
            }

            /* ===== 模型选择栏 ===== */
            #modelLabel {
                font-size: 13px;
                font-weight: bold;
                color: #5c3d2e;
            }

            #modelCombo {
                font-size: 13px;
                padding: 4px 10px;
                background: #fffaf2;
                border: 1px solid #d4b896;
                border-radius: 6px;
                color: #5c3d2e;
                min-width: 160px;
            }
            #modelCombo:hover {
                border-color: #c0392b;
            }
            #modelCombo::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 24px;
                border-left: 1px solid #d4b896;
            }

            #loadBtn {
                font-size: 13px;
                padding: 4px 14px;
                background: #e8d9c8;
                color: #5c3d2e;
                border: 1px solid #d4b896;
                border-radius: 6px;
                font-weight: bold;
            }
            #loadBtn:hover {
                background: #d4b896;
            }

            /* ===== 状态栏 ===== */
            #statusLabel {
                font-size: 13px;
                color: #8b7355;
                padding: 4px 0;
                /* 优先使用 Emoji 字体，防止状态栏图标被削顶 */
                font-family: ".AppleSystemUIFont", "Apple Color Emoji", "Segoe UI Emoji", "STSong", serif; 
            }

            /* ===== 进度条 ===== */
            QProgressBar {
                border: none;
                background-color: #e8d9c8;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #c0392b;
                border-radius: 3px;
            }

            /* ===== 标签页 ===== */
            #mainTabs::pane {
                border: 1px solid #d4b896;
                border-radius: 8px;
                background-color: #fffaf2;
                padding: 0px;
            }
            #mainTabs::tab-bar {
                alignment: center;
            }
            #mainTabs QTabBar::tab {
                background: #e8d9c8;
                color: #5c3d2e;
                padding: 8px 28px;
                margin: 2px 6px 12px 6px; 
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                min-height: 20px;
            }
            #mainTabs QTabBar::tab:selected {
                background: #c0392b;
                color: #ffffff;
            }
            #mainTabs QTabBar::tab:hover:!selected {
                background: #d4b896;
            }

            /* ===== 分组框 ===== */
            #groupBox {
                font-size: 14px;
                font-weight: bold;
                color: #5c3d2e;
                border: 1px solid #d4b896;
                border-radius: 8px;
                margin-top: 16px;        
                padding-top: 18px;       
                padding-bottom: 14px;
                background-color: #fffaf2;
            }
            #groupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 16px;
                top: 0px;
                background-color: #fffaf2; 
                padding: 2px 6px;
                min-height: 24px;          
                font-family: ".AppleSystemUIFont", "Apple Color Emoji", "Segoe UI Emoji", "STSong", serif; 
            }

            /* ===== 输入框 ===== */
            #inputField {
                font-size: 16px;
                padding: 10px 14px;
                border: 2px solid #d4b896;
                border-radius: 8px;
                background-color: #ffffff;
                color: #2c1810;
            }
            #inputField:focus {
                border-color: #c0392b;
            }

            /* ===== 滑块 ===== */
            QSlider::groove:horizontal {
                height: 6px;
                background: #e8d9c8;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 18px;
                height: 18px;
                margin: -6px 0;
                background: #c0392b;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #a93226;
            }
            QSlider::sub-page:horizontal {
                background: #c0392b;
                border-radius: 3px;
            }

            /* ===== 生成按钮 ===== */
            #generateBtn {
                font-size: 16px;
                font-weight: bold;
                padding: 12px;
                background-color: #c0392b;
                color: #ffffff;
                border: none;
                border-radius: 8px;
            }
            #generateBtn:hover {
                background-color: #a93226;
            }
            #generateBtn:pressed {
                background-color: #922b21;
            }
            #generateBtn:disabled {
                background-color: #c0b0a0;
                color: #e0d0c0;
            }

            /* ===== 复制按钮 ===== */
            #copyBtn {
                font-size: 12px;
                padding: 6px 16px;
                background-color: #e8d9c8;
                color: #5c3d2e;
                border: 1px solid #d4b896;
                border-radius: 6px;
                min-height: 20px;
                font-family: ".AppleSystemUIFont", "Apple Color Emoji", "Segoe UI Emoji", "STSong", serif;
            }
            #copyBtn:hover {
                background-color: #d4b896;
            }

            /* ===== 输出区 ===== */
            #outputArea {
                font-size: 18px;
                padding: 8px;            /* 👈 稍微调小内边距，防止渲染体积溢出 */
                margin-bottom: 8px;      /* 👈 核心修复：强制增加底部外边距，把复制按钮稳稳地推下去 */
                border: 1px solid #d4b896;
                border-radius: 8px;
                background-color: #ffffff;
                color: #2c1810;
            }

            /* ===== 滚动条 ===== */
            QScrollBar:vertical {
                width: 8px;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                background: #d4b896;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)


# ========== 入口 ==========

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName('PoetryWriting')

    # 字体回退
    font = QFont('STSong, Songti SC, SimSun, Noto Serif CJK SC')
    font.setPointSize(12)
    app.setFont(font)

    window = PoetryWindow()
    window.show()
    sys.exit(app.exec_())