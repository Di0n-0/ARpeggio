import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QCheckBox, QSlider, QVBoxLayout, QWidget, QLineEdit, QPushButton, QFileDialog

file_path_tab = "ASCII TAB"
ascii_tutor = False
dev_mode = True
show_fretboard = True
mirror_effect = False
slowing_factor = 1
alpha = 1
exit = False


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Settings')
        layout = QVBoxLayout()
        self.dev_mode_checkbox = QCheckBox('Developer Mode')
        self.fretboard_checkbox = QCheckBox('Show Fretboard')
        self.mirror_effect_checkbox = QCheckBox('Mirror Effect')
        self.slowing_factor_label = QLabel('Slowing Factor')
        self.slowing_factor_slider = QSlider(Qt.Orientation.Horizontal)
        self.slowing_factor_slider.setMinimum(0)
        self.slowing_factor_slider.setMaximum(10)
        self.slowing_factor_value_label = QLabel()
        self.alpha_label = QLabel('Alpha Value')
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(10)
        self.alpha_value_label = QLabel()
        self.file_path_input = QLineEdit()
        self.browse_button = QPushButton('Browse')
        self.exit_button = QPushButton('Exit ARpeggio')

        layout.addWidget(self.dev_mode_checkbox)
        layout.addWidget(self.fretboard_checkbox)
        layout.addWidget(self.mirror_effect_checkbox)
        layout.addWidget(self.slowing_factor_label)
        layout.addWidget(self.slowing_factor_slider)
        layout.addWidget(self.slowing_factor_value_label)
        layout.addWidget(self.alpha_label)
        layout.addWidget(self.alpha_slider)
        layout.addWidget(self.alpha_value_label)
        layout.addWidget(self.file_path_input)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.exit_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.exit_button.clicked.connect(self.exit)
        self.browse_button.clicked.connect(self.browse_file)
        self.file_path_input.textChanged.connect(self.file_path_changed)
        self.dev_mode_checkbox.stateChanged.connect(self.dev_mode_changed)
        self.fretboard_checkbox.stateChanged.connect(self.fretboard_changed)
        self.mirror_effect_checkbox.stateChanged.connect(self.mirror_effect_changed)
        self.slowing_factor_slider.valueChanged.connect(self.slowing_factor_changed)
        self.alpha_slider.valueChanged.connect(self.alpha_changed)

        self.file_path_input.setText(file_path_tab)
        self.dev_mode_checkbox.setChecked(dev_mode)
        self.fretboard_checkbox.setChecked(show_fretboard)
        self.mirror_effect_checkbox.setChecked(mirror_effect)
        self.slowing_factor_slider.setValue(slowing_factor)
        self.alpha_slider.setValue(int(alpha))

    def exit(self):
        global exit
        exit = True
        self.close()

    def browse_file(self):
        global file_path_tab
        file_dialog = QFileDialog(self)
        file_path_tab = file_dialog.getOpenFileName(self, 'Select File', '', 'Text Files (*.txt)')[0]
        self.file_path_input.setText(file_path_tab)

    def file_path_changed(self, file_path_tab):
        global ascii_tutor
        if file_path_tab.endswith('.txt'):
            ascii_tutor = True

    def dev_mode_changed(self, state):
        global dev_mode
        dev_mode = state == 2

    def fretboard_changed(self, state):
        global show_fretboard
        show_fretboard = state == 2

    def mirror_effect_changed(self, state):
        global mirror_effect
        mirror_effect = state == 2

    def slowing_factor_changed(self, value):
        global slowing_factor
        slowing_factor = value
        self.slowing_factor_value_label.setText(str(value))

    def alpha_changed(self, value):
        global alpha
        alpha = value
        self.alpha_value_label.setText(str(value))

def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec()
    return file_path_tab, dev_mode, show_fretboard, mirror_effect, ascii_tutor, slowing_factor, alpha, exit
