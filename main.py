import sys
import os
import threading
import html
import json
import time
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QFileDialog, QLabel, QFrame,
    QSlider, QGroupBox, QListWidget, QListWidgetItem, QMenu
)
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QTextCursor, QAction
from llama_cpp import Llama

# 1️⃣ Config file constant
CONFIG_FILE = "config.json"

# --- Signals ---
class ChatSignals(QObject):
    message = Signal(str, str)
    token = Signal(str)
    finished = Signal()

ORANGE_STYLE = """
    QMainWindow { background-color: #121212; }
    QLabel { color: #BBBBBB; font-size: 13px; }
    QGroupBox { color: #FF8C00; font-weight: bold; border: 1px solid #333333; margin-top: 10px; padding: 10px; }
    QPushButton {
        background-color: rgba(255, 140, 0, 0.15);
        color: #FF8C00;
        border: 1px solid rgba(255, 140, 0, 0.3);
        border-radius: 8px;
        padding: 6px 12px;
        font-weight: bold;
    }
    QPushButton:hover { background-color: rgba(255, 140, 0, 0.3); border: 1px solid #FF8C00; }
    QPushButton#primaryBtn { background-color: #FF8C00; color: #000000; }
    QPushButton#primaryBtn:hover { background-color: #FFA500; }
    QPushButton#stopBtn { background-color: rgba(255, 0, 0, 0.15); color: #FF4444; border: 1px solid #FF4444; }
    QPushButton#stopBtn:hover { background-color: rgba(255, 0, 0, 0.3); }
    QTextEdit {
        background-color: #1E1E1E;
        color: #E0E0E0;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 10px;
        font-size: 14px;
    }
    QLineEdit {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
        border: 1px solid #444444;
        border-radius: 20px;
        padding: 10px 15px;
        font-size: 14px;
    }
    QLineEdit:focus { border: 1px solid #FF8C00; }
    QSlider::handle:horizontal { background: #FF8C00; border-radius: 5px; width: 15px; }
    QListWidget {
        background-color: #1E1E1E;
        color: #E0E0E0;
        border: 1px solid #333333;
        border-radius: 8px;
        outline: none;
    }
    QListWidget::item { padding: 10px; border-bottom: 1px solid #252525; }
    QListWidget::item:selected { background-color: #FF8C00; color: black; font-weight: bold; }
"""

class GGUFChat(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Orange AI")
        self.setMinimumSize(1100, 850)
        self.setStyleSheet(ORANGE_STYLE)

        self.llm = None
        self.model_path = None
        self.ai_cursor = None
        self.is_generating = False
        self.stop_requested = False
        self.history = []
        
        self.current_chat_path = None
        self.chats_dir = "chats"
        os.makedirs(self.chats_dir, exist_ok=True)

        self.signals = ChatSignals()
        self.signals.message.connect(self.append_chat)
        self.signals.token.connect(self.append_token)
        self.signals.finished.connect(self.on_generation_finished)

        self.init_ui()
        self.load_config()
        self.refresh_chat_list()

    def save_config(self):
        data = {"last_model": self.model_path}
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_config(self):
        if not os.path.exists(CONFIG_FILE): return
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            last_model = data.get("last_model")
            if last_model and os.path.exists(last_model):
                self.model_path = last_model
                self.model_label.setText(os.path.basename(last_model))
                self.append_chat("System", "Last model remembered. Click 'Load Model' to activate.")
        except Exception: pass

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        sidebar = QVBoxLayout()
        
        new_chat_btn = QPushButton("+ New Chat")
        new_chat_btn.setObjectName("primaryBtn")
        new_chat_btn.clicked.connect(self.new_chat)
        
        sidebar.addWidget(QLabel("SESSIONS"))
        self.chat_list = QListWidget()
        self.chat_list.setMaximumWidth(280)
        self.chat_list.itemClicked.connect(self.load_selected_chat)
        self.chat_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.chat_list.customContextMenuRequested.connect(self.show_context_menu)
        
        sidebar.addWidget(new_chat_btn)
        sidebar.addWidget(self.chat_list)

        model_group = QGroupBox("Model Control")
        model_vbox = QVBoxLayout()
        self.model_label = QLabel("No model loaded")
        self.model_label.setWordWrap(True)
        browse_btn = QPushButton("Browse GGUF")
        self.load_btn = QPushButton("Load Model")
        model_vbox.addWidget(browse_btn)
        model_vbox.addWidget(self.load_btn)
        model_vbox.addWidget(self.model_label)
        model_group.setLayout(model_vbox)
        
        params_group = QGroupBox("Parameters")
        params_vbox = QVBoxLayout()
        self.temp_label = QLabel("Temperature: 0.70")
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 200)
        self.temp_slider.setValue(70)
        self.temp_slider.valueChanged.connect(lambda v: self.temp_label.setText(f"Temperature: {v/100:.2f}"))
        
        self.tokens_label = QLabel("Max Tokens: 512")
        self.tokens_slider = QSlider(Qt.Horizontal)
        self.tokens_slider.setRange(64, 4096)
        self.tokens_slider.setValue(512)
        self.tokens_slider.valueChanged.connect(lambda v: self.tokens_label.setText(f"Max Tokens: {v}"))

        params_vbox.addWidget(self.temp_label)
        params_vbox.addWidget(self.temp_slider)
        params_vbox.addWidget(self.tokens_label)
        params_vbox.addWidget(self.tokens_slider)
        params_group.setLayout(params_vbox)

        export_group = QGroupBox("Export")
        export_vbox = QVBoxLayout()
        save_btn = QPushButton("Save History As...")
        export_vbox.addWidget(save_btn)
        export_group.setLayout(export_vbox)

        sidebar.addWidget(model_group)
        sidebar.addWidget(params_group)
        sidebar.addWidget(export_group)

        chat_vbox = QVBoxLayout()
        self.sys_prompt_edit = QTextEdit()
        self.sys_prompt_edit.setPlaceholderText("System Prompt...")
        self.sys_prompt_edit.setMaximumHeight(60)
        self.sys_prompt_edit.setText("You are a helpful assistant.")

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)

        input_hbox = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type a message...")
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("primaryBtn")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)

        input_hbox.addWidget(self.input_box)
        input_hbox.addWidget(self.send_btn)
        input_hbox.addWidget(self.stop_btn)

        chat_vbox.addWidget(QLabel("SYSTEM PROMPT"))
        chat_vbox.addWidget(self.sys_prompt_edit)
        chat_vbox.addWidget(self.chat_area)
        chat_vbox.addLayout(input_hbox)

        main_layout.addLayout(sidebar, 1)
        main_layout.addLayout(chat_vbox, 3)

        browse_btn.clicked.connect(self.browse_model)
        self.load_btn.clicked.connect(self.load_model)
        self.send_btn.clicked.connect(self.send_message)
        self.input_box.returnPressed.connect(self.send_message)
        self.stop_btn.clicked.connect(self.request_stop)
        save_btn.clicked.connect(self.save_history)

    def generate_chat_title(self):
        for msg in self.history:
            if msg["sender"] == "You":
                clean = msg["message"].strip()[:35]
                return clean if clean else "New Chat"
        return "New Chat"

    def save_current_chat(self):
        if not self.history: return
        title = self.generate_chat_title()
        if not self.current_chat_path:
            safe_title = title.replace(" ", "_").replace("/", "_")[:40]
            ts = int(time.time())
            self.current_chat_path = os.path.join(self.chats_dir, f"{ts}_{safe_title}.json")

        data = {
            "version": 1, "title": title,
            "system": self.sys_prompt_edit.toPlainText(),
            "messages": self.history
        }
        with open(self.current_chat_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.refresh_chat_list()

    def refresh_chat_list(self):
        self.chat_list.blockSignals(True)
        self.chat_list.clear()
        if not os.path.exists(self.chats_dir):
            self.chat_list.blockSignals(False)
            return
        files = [f for f in os.listdir(self.chats_dir) if f.endswith(".json")]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.chats_dir, x)), reverse=True)
        for file in files:
            display_name = file.split("_", 1)[-1].replace(".json", "").replace("_", " ")
            item = QListWidgetItem(display_name)
            item.setData(Qt.UserRole, os.path.join(self.chats_dir, file))
            self.chat_list.addItem(item)
        self.chat_list.blockSignals(False)

    def load_selected_chat(self, item):
        path = item.data(Qt.UserRole)
        self.save_current_chat()
        if not os.path.exists(path): return
        self.current_chat_path = path
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chat_area.clear()
        self.history = []
        self.sys_prompt_edit.setText(data.get("system", "You are a helpful assistant."))
        for msg in data.get("messages", []):
            self.append_chat(msg["sender"], msg["message"])

    def new_chat(self):
        self.save_current_chat()
        self.chat_area.clear()
        self.history = []
        self.current_chat_path = None
        self.sys_prompt_edit.setText("You are a helpful assistant.")
        self.append_chat("System", "Ready for a new conversation.")

    def show_context_menu(self, pos):
        item = self.chat_list.itemAt(pos)
        if not item: return
        menu = QMenu()
        delete_action = QAction("Delete Chat", self)
        delete_action.triggered.connect(lambda: self.delete_chat_file(item))
        menu.addAction(delete_action)
        menu.exec(self.chat_list.mapToGlobal(pos))

    def delete_chat_file(self, item):
        path = item.data(Qt.UserRole)
        if os.path.exists(path): os.remove(path)
        if self.current_chat_path == path: self.new_chat()
        self.refresh_chat_list()

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF", "", "GGUF Files (*.gguf)")
        if path:
            self.model_path = path
            self.model_label.setText(os.path.basename(path))

    def load_model(self):
        if not self.model_path: return
        self.append_chat("System", "Loading model into memory...")
        def load():
            try:
                if self.llm is not None:
                    del self.llm
                    self.llm = None
                
                ctx = max(2048, self.tokens_slider.value() + 512)
                self.llm = Llama(model_path=self.model_path, n_ctx=ctx)
                
                self.save_config()
                self.signals.message.emit("System", "Model ready!")
            except Exception as e:
                self.signals.message.emit("System", f"Load Error: {str(e)}")
        threading.Thread(target=load, daemon=True).start()

    def request_stop(self):
        self.stop_requested = True
        self.stop_btn.setEnabled(False)

    # 🔴 UPDATED: Context Capped Prompt Building
    def build_prompt(self, max_turns=10):
        prompt = f"<|system|>\n{self.sys_prompt_edit.toPlainText()}\n"
        # We slice the history to keep only the last X turns (user + assistant)
        for msg in self.history[-max_turns*2:]:
            role = "user" if msg["sender"] == "You" else "assistant"
            prompt += f"<|{role}|>\n{msg['message']}\n"
        prompt += "<|assistant|>\n"
        return prompt

    def send_message(self):
        if not self.llm or self.is_generating: return
        text = self.input_box.text().strip()
        if not text: return

        self.append_chat("You", text)
        self.input_box.clear()
        self.is_generating, self.stop_requested = True, False
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        def generate():
            full_ai_response = ""
            try:
                # 🔧 STEP 2: Create streaming cursor without touching history
                self.signals.message.emit("__STREAM__", "")
                
                prompt = self.build_prompt()
                
                stream = self.llm(
                    prompt, 
                    max_tokens=self.tokens_slider.value(), 
                    temperature=self.temp_slider.value()/100.0, 
                    stream=True, 
                    stop=["<|user|>", "</s>", "<|system|>"]
                )

                for chunk in stream:
                    if self.stop_requested: break
                    token = chunk["choices"][0]["text"]
                    full_ai_response += token
                    self.signals.token.emit(token)
                
            except Exception as e:
                self.signals.message.emit("System", f"Error: {str(e)}")
            finally:
                # 🔧 STEP 3: Append AI message after generation completes
                if full_ai_response.strip():
                    self.history.append({
                        "sender": "AI",
                        "message": full_ai_response
                    })
                self.signals.finished.emit()

        threading.Thread(target=generate, daemon=True).start()

    def on_generation_finished(self):
        self.is_generating = False
        self.ai_cursor = None
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_current_chat()

    def append_token(self, token):
        if self.ai_cursor:
            self.ai_cursor.insertText(token)
            self.chat_area.verticalScrollBar().setValue(self.chat_area.verticalScrollBar().maximum())

    # 🔧 UPDATED: append_chat with specialized streaming logic
    def append_chat(self, sender, message):
        if sender == "__STREAM__":
            self.chat_area.append(
                '<b><span style="color:#BBBBBB;">AI:</span></b> '
            )
            self.ai_cursor = self.chat_area.textCursor()
            self.ai_cursor.movePosition(QTextCursor.End)
            return

        if sender != "System":
            self.history.append({"sender": sender, "message": message})

        safe_msg = html.escape(message)
        color = "#FF8C00" if sender == "You" else "#BBBBBB"
        if sender == "System":
            color = "#555555"

        self.chat_area.append(
            f'<b><span style="color:{color};">{sender}:</span></b> {safe_msg}'
        )
        self.chat_area.verticalScrollBar().setValue(self.chat_area.verticalScrollBar().maximum())

    def save_history(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON", "", "JSON Files (*.json)")
        if not path: return
        if not path.lower().endswith(".json"): path += ".json"
        data = {
            "version": 1, "system": self.sys_prompt_edit.toPlainText(), "messages": self.history
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GGUFChat()
    window.show()
    sys.exit(app.exec())