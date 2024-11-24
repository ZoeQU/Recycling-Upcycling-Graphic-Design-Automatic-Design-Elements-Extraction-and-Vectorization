# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets, QtSvg
from PyQt5.QtSvg import QGraphicsSvgItem
from PyQt5.QtWidgets import (QApplication, QGraphicsScene,QVBoxLayout, 
                             QMainWindow, QWidget,
                             QPushButton, QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap
import sys
import cv2
import os
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  #type: ignore


from setting import (savePath, imgPath, elementPath, processPath, 
                     keepPath, svgPath, colorrefPath)

from utils import image_processing
from src.extraction.extractor import extractor
from src.vectorization.vectorizator import vectorize_design_element
from src.generation.motif_generator import generate_motif_pattern
from src.generation.check_generator import generate_check_pattern
from src.generation.stripe_generator import generate_stripe_pattern

import warnings
warnings.filterwarnings("ignore")

def mkfolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

for path in [savePath, imgPath, elementPath, processPath, keepPath, svgPath]:
    mkfolder(path)


class Ui_MainWindow(object):
    def __init__(self) -> None:
        super().__init__()
        self.element_pathes = None
        self.color_block_path = None
        self.svg_name = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(766, 444)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("ui/qu_icon_16x16.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Load "input raster image" button
        self.LoadImageInput = QtWidgets.QPushButton(self.centralwidget)
        self.LoadImageInput.setGeometry(QtCore.QRect(150, 30, 70, 26))
        self.LoadImageInput.setObjectName("LoadImageInput")
        self.LoadImageInput.setText("Load")
        self.LoadImageInput.clicked.connect(self.load_input_image)

        # Load "input raster image" text
        self.ImageInput = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.ImageInput.setGeometry(QtCore.QRect(30, 30, 110, 26))
        self.ImageInput.setObjectName("ImageInput")

        # "Input raster image" view
        self.ImageInputView = QtWidgets.QGraphicsView(self.centralwidget)
        self.ImageInputView.setGeometry(QtCore.QRect(30, 65, 190, 180))
        self.ImageInputView.setObjectName("InputImageView")


        # Vertical line (QFrame) to separate the input area
        self.VerticalLine = QtWidgets.QFrame(self.centralwidget)
        self.VerticalLine.setGeometry(QtCore.QRect(240, 55, 12, 360)) 
        self.VerticalLine.setFrameShape(QtWidgets.QFrame.VLine)  
        self.VerticalLine.setFrameShadow(QtWidgets.QFrame.Sunken)

        # Show "Output svg image" text
        self.SvgOutput = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.SvgOutput.setGeometry(QtCore.QRect(410, 30, 200, 26))
        self.SvgOutput.setObjectName("SvgOutput")
        self.SvgOutput.setPlainText("Output SVG Image") 
        self.SvgOutput.setStyleSheet("border: none; background: transparent;")
        self.SvgOutput.setReadOnly(True) 

        # Show "Hint" message
        self.Hint = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.Hint.setGeometry(QtCore.QRect(30, 255, 200, 55))
        self.Hint.setObjectName("SvgOutput")
        self.Hint.setPlainText("Hint: The higher the domain value, the more design elements are retained.") 
        self.Hint.setStyleSheet("border: none; background: transparent;")
        self.Hint.setReadOnly(True) 
        
        # dynamically generated for small windows - ScrollArea
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(260, 60, 460, 350))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 460, 350))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        
        self.gridLayout = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        # Horizontal scrollbar
        self.ScrollBar = QtWidgets.QScrollBar(self.centralwidget)
        self.ScrollBar.setGeometry(QtCore.QRect(40, 310, 140, 20))  # Position below the download button
        self.ScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.ScrollBar.setMinimum(-10)  # Set the scroll bar's minimum value
        self.ScrollBar.setMaximum(10)   # Set the scroll bar's maximum value
        self.ScrollBar.setValue(0)     # Default value
        self.ScrollBar.setSingleStep(1)  # Smallest step (mapped to 0.1)
        
        # Label for scrollbar value
        self.ScrollBarValueLabel = QtWidgets.QLabel(self.centralwidget)
        self.ScrollBarValueLabel.setGeometry(QtCore.QRect(193, 310, 60, 20))  # Position next to the scrollbar
        self.ScrollBarValueLabel.setObjectName("ScrollBarValueLabel")
        self.ScrollBarValueLabel.setText("0.0")  # Default text

        # Connect scrollbar value change to the update function
        self.ScrollBar.valueChanged.connect(self.update_scrollbar_value)

        # Generate button
        self.GenerateButton = QtWidgets.QPushButton(self.centralwidget)
        self.GenerateButton.setGeometry(QtCore.QRect(40, 345, 170, 26))
        self.GenerateButton.setObjectName("GenerateButton")
        self.GenerateButton.clicked.connect(self.generate_button_click)

        # Download button
        self.DownloadButton = QtWidgets.QPushButton(self.centralwidget)
        self.DownloadButton.setGeometry(QtCore.QRect(40, 380, 170, 26))
        self.DownloadButton.setObjectName("DownloadButton")
        self.DownloadButton.clicked.connect(self.download_all_button_click)

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Automatic Design Elements Extraction and Vectorization"))
        self.DownloadButton.setText(_translate("MainWindow", "Download"))
        self.LoadImageInput.setText(_translate("MainWindow", "Load"))
        self.ImageInput.setPlainText(_translate("MainWindow", "Input Image"))
        self.GenerateButton.setText(_translate("MainWindow", "Extracte"))


    def update_scrollbar_value(self):
        value = self.ScrollBar.value()  
        float_value = value / 10.0 
        self.thre = float_value 
        self.ScrollBarValueLabel.setText(f"{float_value:.1f}")

    def download_button_click(self):
        print("Download button被click了")
        try:
            # Let the user choose a folder (pass `self` as the parent)
            folder_path = QFileDialog.getExistingDirectory(None, "Select Folder to Save Images", "/home/zoe/ResearchProjects/DesignGenerationVector/data/temp/temp_save")
            if not folder_path:
                QMessageBox.warning(None, "Warning", "No folder selected!", QMessageBox.Ok)
                return

            # Save the PNG and SVG files
            self.save_file(self.color_block_path, folder_path)
            self.save_file(self.svg_name, folder_path)

            # Show success message
            QMessageBox.information(None, "Success", f"Both images have been saved to:\n{folder_path}", QMessageBox.Ok)

        except Exception as e:
            # Display error message if something goes wrong
            QMessageBox.critical(None, "Error", f"An error occurred while saving files: {str(e)}", QMessageBox.Ok)

    def save_file(self, source_path, target_folder):
        """
        Save a file from source_path to the target folder.
        """
        if not os.path.exists(source_path):
            QMessageBox.warning(None, "Error", f"File not found: {source_path}", QMessageBox.Ok)
            return False

        target_path = os.path.join(target_folder, os.path.basename(source_path))
        try:
            with open(source_path, "rb") as src_file:
                with open(target_path, "wb") as target_file:
                    target_file.write(src_file.read())
            return True
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to save file: {str(e)}", QMessageBox.Ok)
            return False

    # ================================================================
    def generate_button_click(self):
        print("Extracte button clicked")
        bk_color, keep = extractor(self.input_design_image, self.name, self.thre, visualization=True)
        element_pathes, _ = vectorize_design_element(self.name, keep, bk_color, None, visualization=True)
        self.svg_data_list = element_pathes
        self.display_svg_outputs(self.svg_data_list)


    def display_svg_outputs(self, svg_data_list):
        # Dynamically generate small windows for svg and corresponding download button
        for i in reversed(range(self.gridLayout.count())):
            widget = self.gridLayout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        for i, svg_path in enumerate(svg_data_list):
            row, col = divmod(i, 2)  # 2 window in one row
            
            graphics_view = QtWidgets.QGraphicsView(self.scrollAreaWidgetContents)
            graphics_view.setMinimumSize(200, 150)  
            graphics_view.setStyleSheet("border: 1px solid gray;")

            scene = QtWidgets.QGraphicsScene(graphics_view)
            graphics_view.setScene(scene)

            if os.path.exists(svg_path):
                svg_item = QtSvg.QGraphicsSvgItem(svg_path)
                scene.addItem(svg_item)

                scene.setSceneRect(svg_item.boundingRect())

                graphics_view.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            else:
                placeholder_label = QtWidgets.QLabel("SVG not found", graphics_view)
                placeholder_label.setAlignment(QtCore.Qt.AlignCenter)
                placeholder_label.setStyleSheet("color: red;")
                placeholder_label.setGeometry(0, 0, 200, 150)

            # 2. download button
            download_button = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
            download_button.setText(f"Download {os.path.basename(svg_path)}")
            download_button.clicked.connect(lambda _, path=svg_path: self.download_specific_svg(path))

            # 3. add them into UI
            self.gridLayout.addWidget(graphics_view, row * 2, col)
            self.gridLayout.addWidget(download_button, row * 2 + 1, col)


    def download_specific_svg(self, svg_path):
        print(f"Downloading SVG" + str(svg_path))
 
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder to Save Images", 
                                                       "/home/zoe/ResearchProjects/DesignGenerationVector/data/temp/temp_save")
      
        save_path = os.path.join(folder_path, os.path.basename(svg_path))

        with open(svg_path, "rb") as src_file:
            with open(save_path, "wb") as target_file:
                target_file.write(src_file.read())
            
        QtWidgets.QMessageBox.information(None, "Success", f"SVG saved as {save_path}")

    def download_all_button_click(self):
        print("Download all SVGs button clicked")
        try:
            # Let the user choose a folder (pass `self` as the parent)
            folder_path = QFileDialog.getExistingDirectory(None, "Select Folder to Save Images", "/home/zoe/ResearchProjects/DesignGenerationVector/data/temp/temp_save")
            if not folder_path:
                QMessageBox.warning(None, "Warning", "No folder selected!", QMessageBox.Ok)
                return
            
            for svg in self.svg_data_list:
                print(svg)
                self.save_file(svg, folder_path)

            # Show success message
            QMessageBox.information(None, "Success", f"Both images have been saved to:\n{folder_path}", QMessageBox.Ok)

        except Exception as e:
            # Display error message if something goes wrong
            QMessageBox.critical(None, "Error", f"An error occurred while saving files: {str(e)}", QMessageBox.Ok)



    # ================================================================    
    def regenerate_button_click(self):
        print("ReGenerate button被click了")
        self.input_pattern_type = self.PatternTypecomboBox.currentText()
        self.input_color_number = self.ColorSpinBox.value()
                
        if self.input_pattern_type == "Check":
            svg_name, color_block_path = generate_check_pattern(self.input_color_image, num=self.input_color_number)
            self.color_block_path = color_block_path
            self.display_gengrate_images(svg_name, color_block_path)
            
        elif self.input_pattern_type == "Motif":
            for element_path in self.element_pathes:
                svg_name = generate_motif_pattern(element_path)

                self.display_gengrate_images(svg_name, self.color_block_path)

        else:
            svg_name, color_block_path = generate_stripe_pattern(self.input_color_image, num=self.input_color_number)
            self.display_gengrate_images(svg_name, color_block_path)


    def load_svg(self):
        # Open file dialog to select SVG
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(None, "Select SVG File", "/home/zoe/ResearchProjects/DesignGenerationVector/resources",
                                                  "SVG Files (*.svg)", options=options)

        if fileName:
            self.DesignElement.setPlainText(fileName.split("/")[-1])
            self.display_svg(self.DesignElementView, fileName)
            # self.LoadDesignElement.setEnabled(False)  # Disable button after loading


    def load_color_reference_image(self):
        # Open file dialog to select image
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(None, "Select Image File", "/home/zoe/ResearchProjects/DesignGenerationVector/data/color_ref",
                                                  "Image Files (*.png *.jpg *.jpeg)", options=options)

        if fileName:
            self.ColorReferenceImage.setPlainText(fileName.split("/")[-1])
            self.display_input_image(self.ColorRefView, fileName)
            self.input_color_image = cv2.imread(fileName, cv2.IMREAD_COLOR)


    def load_input_image(self):
        # Open file dialog to select image
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(None, "Select Image File", "/home/zoe/ResearchProjects/DesignGenerationVector/data/input",
                                                  "Image Files (*.png *.jpg *.jpeg)", options=options)

        if fileName:
            # self.ColorReferenceImage.setPlainText(fileName.split("/")[-1])
            self.display_input_image(self.ImageInputView, fileName)
            self.input_design_image = cv2.imread(fileName, cv2.IMREAD_COLOR)
            self.name = fileName.split("/")[-1].split(".")[0]


    def display_svg(self, view, file_path):
        scene = QGraphicsScene()
        svg_item = QGraphicsSvgItem(file_path)

        # Get dimensions of the view and SVG
        view_width = view.width()
        view_height = view.height()
        svg_rect = svg_item.boundingRect()

        # Calculate scale factor while maintaining aspect ratio
        scale_factor = min(view_width / svg_rect.width(), view_height / svg_rect.height())
        svg_item.setScale(scale_factor)

        # Center the SVG in the view
        svg_item.setPos(
            (view_width - svg_rect.width() * scale_factor) / 2,
            (view_height - svg_rect.height() * scale_factor) / 2
        )

        scene.addItem(svg_item)
        view.setScene(scene)


    def display_input_image(self, view, file_path):
        scene = QGraphicsScene()
        pixmap = QPixmap(file_path)

        # Get dimensions of the view and image
        view_width = view.width()
        view_height = view.height()

        # Scale the pixmap to fill the view while maintaining the aspect ratio
        pixmap = pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatioByExpanding)

        # Center the image in the view
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        item.setPos(
            (view_width - pixmap.width()) / 2,
            (view_height - pixmap.height()) / 2
        )

        scene.addItem(item)
        view.setScene(scene)


    def display_gengrate_images(self, svg_name, color_pallete_name):
        # Load and display the SVG
        svg_scene = QtWidgets.QGraphicsScene()
        self.svg_item = QtSvg.QGraphicsSvgItem(svg_name) 
        svg_scene.addItem(self.svg_item)
        self.graphicsView.setScene(svg_scene)  # Set the scene in the view
        # Fit the view to the initial size
        self.graphicsView.fitInView(svg_scene.sceneRect(),  QtCore.Qt.KeepAspectRatioByExpanding)

        # # switch to display png
        # svg2png_scene = QtWidgets.QGraphicsScene()
        # svg2png_image = QtGui.QPixmap(svg_name)
        # svg2png_scene.addPixmap(svg2png_image)
        # self.graphicsView.setScene(svg2png_scene)
        # self.graphicsView.fitInView(svg2png_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

        # Load and display the PNG
        png_scene = QtWidgets.QGraphicsScene()
        png_image = QtGui.QPixmap(color_pallete_name) 
        png_scene.addPixmap(png_image)
        self.ColorPaletteView.setScene(png_scene)


    def load_obj_model(self, filename):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(filename)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()

        self.ThreeDWidget.GetRenderWindow().Render()
        self.iren.Initialize()


    def scale_image(self):
        # Scale the SVG based on the scrollbar value
        if self.svg_item is not None:
            # Ensure the scale factor starts at 1 for the initial size
            min_scale = 1.0
            max_scale = 10.0  # Adjust as needed for maximum scaling
            scale_factor = min_scale + (self.ScaleScrollBar.value() - self.ScaleScrollBar.minimum()) / (self.ScaleScrollBar.maximum() - self.ScaleScrollBar.minimum()) * (max_scale - min_scale)
            
            self.svg_item.setScale(scale_factor)
            self.graphicsView.setSceneRect(self.svg_item.boundingRect())
            self.graphicsView.fitInView(self.graphicsView.sceneRect(), QtCore.Qt.KeepAspectRatioByExpanding)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
