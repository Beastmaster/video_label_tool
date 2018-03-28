'''
Implement UI logic
'''
import time
import numpy as np
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from ui import Ui_MainWindow

from load_labelled_file import *
from overlay_image import *
from config import decode_config
from apply_modify import apply_modify,draw_points_on_label

class MainWindow(Ui_MainWindow,QMainWindow):
    def __init__(self):
        Ui_MainWindow.__init__(self)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.disp_canvas.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label_name_value_listWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        
        ## resources
        self.refresh_timer = QTimer()
        self.rtmp_timer = QTimer()
        self.image_file_list = []
        self.label_file_list = []
        self.image = None
        self.label = None
        self.frame_cnt = 0
        self.frame_cnt_ahead = 0
        self.rtmp_link = ""

        self.refresh_timeout = 2 # millisecond
        self.webcam = None
        self.boverlay_disp = False

        self.scale_rate = 1.0
        self.leftButtonDown = False
        self.pixmap_origin = None
        self.rect_origin = None
        self.rect_end = None
        self.pixmap = None
        self.drawn_points = []
        self.erase_points = []
        self.brush_size = self.brush_size_spinBox.value()

        self.orginal_image_dir = None
        self.segemtation_image_dir = None
        self.aug_background_image_file = None
        self.aug_output_folder = None

        self.default_label_value = "stand"
        self.labeling_info = [] # example: [ (0,100,"value") , (200,300,"value") ] 

        ## connect
        self.signal_connection()

    def read_config(self):
        '''
        Read configuration file after .show() function called (on main.py)
        '''
        configs = decode_config("config.json")
        if "image_dir" in configs.keys():
            image_file_path = configs["image_dir"]
            self.image_file_list = walk_image_dir(image_file_path,"*.jpg")
            self.print_info("Totally {} file found".format(len(self.image_file_list)))
            # progress bar
            self.progress_bar.setDisabled(False)
            self.progress_bar.setMaximum(len(self.image_file_list))
            # draw first frame
            try:
                img = cv2.cvtColor(cv2.imread(self.image_file_list[0]),cv2.COLOR_BGR2RGB)
                self.show_picture_on_label(img,self.disp_canvas)
            except Exception as e:
                self.print_info("Cannot play video"+str(e),clear=False)

        if "label_dir" in configs.keys():
            label_file_path = configs["label_dir"]
            self.label_file_list = walk_image_dir(label_file_path,"*.jpg")
            if len(self.label_file_list) != len(self.image_file_list):
                self.print_info("Image file and label file do not match..")
        if "refresh_timeout" in configs.keys():
            self.refresh_timeout = configs['refresh_timeout']
        if "rtmp_link" in configs.keys():
            self.rtmp_link = configs['rtmp_link']
        
        
    def signal_connection(self):
        self.actionOpen.triggered.connect(self.on_action_open)
        self.actionSelect_Dir.triggered.connect(self.on_action_SelectDir)
        self.actionSelect_Overlay.triggered.connect(self.on_action_SelectLabelDir)
        self.play_btn.pressed.connect(self.on_play_clicked)
        self.forward_btn.pressed.connect(self.on_forward_btn)
        self.backward_btn.pressed.connect(self.on_backward_btn)
        self.rtmp_btn.pressed.connect(self.on_rtmp_btn)
        self.save_rtmp2video_btn.pressed.connect(self.on_save_rtmp2video_btn)
        self.overlay_check.toggled.connect(self.on_overlay_check_changed)
        self.refresh_timer.timeout.connect(self.on_refresh_timeout)
        self.rtmp_timer.timeout.connect(self.on_rtmp_timer)
        self.progress_bar.sliderMoved.connect(self.on_slider_move)
        self.progress_bar_ahead.sliderMoved.connect(self.on_slider_ahead_move)
        self.progress_bar.setDisabled(True)
        self.check_block.toggled.connect(self.on_check_block)
        self.check_brush.toggled.connect(self.on_check_brush)
        self.brush_size_spinBox.valueChanged.connect(self.on_brushsize_spin)
        self.function_TabWidget.currentChanged.connect(self.on_functionTabWidget_change)
        self.btn_generate_label_file.pressed.connect(self.on_generate_label_file)
        self.btn_confirm_labeling.pressed.connect(self.on_confirm_labeling)
        self.btn_add_new_label_value.pressed.connect(self.on_add_new_label_value)
        self.check_enable_labeling.toggled.connect(self.on_check_enable_labeling)
        self.label_name_value_listWidget.customContextMenuRequested.connect(self.rightClickContextMenu)
        self.btn_original_img.pressed.connect(self.on_btn_original_imgdir)
        self.btn_seg_mask.pressed.connect(self.on_btn_seg_mask_dir)
        self.btn_sel_background.pressed.connect(self.on_btn_sel_background)
        self.btn_output_folder.pressed.connect(self.on_btn_output_dir)
        self.btn_start_augmentation.pressed.connect(self.on_btn_start_augment)
    
    def show_picture_on_label(self,img,label):
        if len(img.shape)<3:
            img = np.stack((img,img,img),axis=-1)
        height, width, bytesPerComponent = img.shape
        bytesPerLine = bytesPerComponent * width
        qimage = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        width_scale = label.width()/width
        height_scale = label.height()/height
        self.scale_rate = min(width_scale,height_scale)
        self.pixmap = QPixmap.fromImage(qimage) #.scaled(self.scale_rate*width,self.scale_rate*height)
        self.pixmap_tmp = self.pixmap.copy()
        self.pixmap_origin = [self.pixmap.physicalDpiX(),self.pixmap.physicalDpiY()]
        label.setPixmap(self.pixmap)
    
    def print_info(self,info,clear = False):
        time_str = time.strftime("%a, %d %b %Y %H:%M:%S\n", time.gmtime())
        if clear:
            self.info_box.setText(time_str+'\n'+str(info))
        else:
            self.info_box.append("\n"+time_str+'\n'+str(info))
    def reset_display_tab(self):
        '''
        GUI function: reset all checks and radios
        '''
        self.overlay_check.setChecked(False)
        self.check_block.setChecked(False)
        self.check_brush.setChecked(False)
        self.check_enable_labeling.setChecked(False)


    def apply_block_modify(self):
        """
        Apply block selection and save to file
        """
        if self.label is None:
            self.print_info("Not in mask display mode...",clear=True)
            return 
        if self.check_block.isChecked():
            if self.radio_block_del.isChecked():
                self.label = apply_modify(self.label,self.rect_origin,self.rect_end,mode="delete")
            if self.radio_block_reserve.isChecked():
                self.label = apply_modify(self.label,self.rect_origin,self.rect_end,mode="reserve")
            # save label file
            self.print_info("writing to {}".format(self.label_file_list[self.frame_cnt]),clear=True)
            cv2.imwrite(self.label_file_list[self.frame_cnt],self.label)
    
    def apply_drawn_modify(self):
        '''
        Apply drawn points on opencv image and save
        '''
        if self.label is None:
            self.print_info("Not in mask display mode...")
            return
        if self.check_brush.isChecked():
            mod_label = draw_points_on_label(self.label,self.drawn_points)
            mod_label = draw_points_on_label(mod_label,self.erase_points,mode='del')
            self.print_info("writing to {}".format(self.label_file_list[self.frame_cnt]),clear=True)
            cv2.imwrite(self.label_file_list[self.frame_cnt],mod_label)
       

    def refresh_view(self, pos = None):
        '''     '''
        # clear print brush buffer
        self.drawn_points = []
        self.erase_points = []

        if pos is not None:
            self.frame_cnt = pos
        if self.frame_cnt >= len(self.image_file_list):
            self.refresh_timer.stop()
            self.frame_cnt = 0
            self.print_info("Play video finished")
            return
        filename = self.image_file_list[self.frame_cnt]
        self.img = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)
        if self.boverlay_disp and len(self.image_file_list)==len(self.label_file_list):
            labelname = self.label_file_list[self.frame_cnt]
            self.label = cv2.imread(labelname,2)
            overalyed_img = overlay_image(self.img,self.label)
            self.show_picture_on_label(overalyed_img,self.disp_canvas)
        else:
            self.show_picture_on_label(self.img,self.disp_canvas)
        self.progress_bar.setSliderPosition(self.frame_cnt)
        self.frame_cnt_label.setText(str(self.frame_cnt))

    ############ fucntion tab########
    def on_functionTabWidget_change(self,state):
        self.reset_display_tab()

    ########## slot functions ############
    def on_btn1_clicked(self):
        QMessageBox.information(self,"Welcome!","OK")

    def on_slider_move(self,pos):
        self.apply_block_modify()
        self.frame_cnt = pos
        if self.frame_cnt >= len(self.image_file_list):
            self.refresh_timer.stop()
            self.frame_cnt = 0
            self.print_info("Play video finished")
            return
        self.frame_cnt_label.setText(str(self.frame_cnt))
        filename = self.image_file_list[self.frame_cnt]
        self.img = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)
        if self.boverlay_disp and len(self.image_file_list)==len(self.label_file_list):
            labelname = self.label_file_list[self.frame_cnt]
            self.label = cv2.imread(labelname,-1)
            overalyed_img = overlay_image(self.img,self.label)
            self.show_picture_on_label(overalyed_img,self.disp_canvas)
        else:
            self.show_picture_on_label(self.img,self.disp_canvas)

    def on_slider_ahead_move(self,pos):
        self.frame_cnt_ahead = pos
        if self.frame_cnt_ahead >= len(self.image_file_list):
            self.refresh_timer.stop()
            self.frame_cnt_ahead = 0
            return
        self.frame_cnt_ahead_label.setText(str(self.frame_cnt_ahead))
        filename = self.image_file_list[self.frame_cnt_ahead]
        self.img = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)
        if self.boverlay_disp and len(self.image_file_list)==len(self.label_file_list):
            labelname = self.label_file_list[self.frame_cnt_ahead]
            self.label = cv2.imread(labelname,-1)
            overalyed_img = overlay_image(self.img,self.label)
            self.show_picture_on_label(overalyed_img,self.disp_canvas)
        else:
            self.show_picture_on_label(self.img,self.disp_canvas)


    def on_overlay_check_changed(self):
        self.boverlay_disp = self.overlay_check.isChecked()
        self.refresh_view()


    def on_action_open(self):
        filename = QFileDialog.getOpenFileName(self,"openfile","./")[0]
        if filename=='':
            self.print_info("File name selection fail")
            return
        self.img = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)
        if img is None:
            self.print_info("Image file: {} invalid!".format(filename))
            return 
        self.show_picture_on_label(img,self.disp_canvas)
    
    def on_action_SelectDir(self):
        path = QFileDialog.getExistingDirectory(self,"select dir",'./')
        if path=='':
            return
        self.image_file_list = walk_image_dir(path,"*.jpg")
        self.print_info("Totally {} file found".format(len(self.image_file_list)))

        # progress bar
        self.progress_bar.setDisabled(False)
        self.progress_bar.setMaximum(len(self.image_file_list))
        # draw first frame
        try:
            img = cv2.cvtColor(cv2.imread(self.image_file_list[0]),cv2.COLOR_BGR2RGB)
            self.show_picture_on_label(img,self.disp_canvas)
        except Exception as e:
            self.print_info("Cannot play video"+str(e),clear=False)
        
    def on_action_SelectLabelDir(self):
        path = QFileDialog.getExistingDirectory(self,"select dir",'./')
        if path=='':
            return
        self.label_file_list = walk_image_dir(path,"*.jpg")

    def on_play_clicked(self):
        if self.image_file_list is None:
            return
        if self.overlay_check.isChecked():
            if len(self.image_file_list) != len(self.label_file_list):
                self.boverlay_disp = False
                self.print_info("image file xxx  label file")
                self.overlay_check.setChecked(False)
            else:
                self.boverlay_disp = True
        else:
            self.boverlay_disp = False

        if self.play_btn.text().upper() =='PLAY':
            self.refresh_timer.start(self.refresh_timeout)
            self.play_btn.setText("Pause")
        else:
            self.refresh_timer.stop()
            self.play_btn.setText("Play")
            
    def on_refresh_timeout(self):
        self.apply_block_modify()
        self.apply_drawn_modify()
        self.frame_cnt+=1
        self.refresh_view()

    def on_forward_btn(self):
        self.apply_block_modify()
        self.apply_drawn_modify()
        self.frame_cnt += 1
        self.refresh_view(pos=self.frame_cnt)
        
    def on_backward_btn(self):
        self.frame_cnt -= 1
        if self.frame_cnt<0:
            self.frame_cnt=0
            return 
        self.refresh_view(pos=self.frame_cnt)
        
    
    def on_check_block(self, state):
        if state:
            self.disp_canvas.setMouseTracking(True)
            self.radio_block_del.setEnabled(True)
            self.radio_block_reserve.setEnabled(True)
        else:
            self.disp_canvas.setMouseTracking(False)
            self.radio_block_del.setChecked(False)
            self.radio_block_del.setEnabled(False)
            self.radio_block_reserve.setChecked(False)
            self.radio_block_reserve.setEnabled(False)
    
    def on_check_brush(self,state):
        if  state:
            self.radio_brush_add.setEnabled(True)
            self.radio_brush_del.setEnabled(True)
        else:
            self.radio_brush_add.setChecked(False)
            self.radio_brush_del.setChecked(False)
            self.radio_brush_add.setEnabled(False)
            self.radio_brush_del.setEnabled(False)
    def on_brushsize_spin(self,sz):
        self.brush_size = sz #self.brush_size_spinBox.value()

    def mouseMoveEvent(self, event):
        if self.leftButtonDown:
            self.info_box.setText("{},{}".format(event.x(),event.y()))
            npos = self.disp_canvas.mapFrom(self,event.pos())
            if self.check_block.isChecked():
                self.pixmap_tmp = self.pixmap.copy()
                self.rect_end = (npos.x(),npos.y())    
            if self.check_brush.isChecked():
                if self.radio_brush_add.isChecked():
                    self.drawn_points.append((npos.x(),npos.y(),self.brush_size_spinBox.value()))
                else:
                    ele = (npos.x(),npos.y(),self.brush_size_spinBox.value())
                    if ele in self.drawn_points:
                        self.drawn_points.remove(ele)
                    else:
                        self.erase_points.append(ele)
            self.update()

    def mousePressEvent(self,event):
        if event.button() == Qt.LeftButton:
            self.leftButtonDown = True
            npos = self.disp_canvas.mapFrom(self,event.pos())
            if self.check_block.isChecked():
                self.rect_origin = (npos.x(),npos.y())
                self.pixmap_tmp = self.pixmap.copy()
    
    def mousetReleaseEvent(self,event):
        if event.button() == Qt.LeftButton:
            self.leftButtonDown = False

    def paintEvent(self, event):
        if self.check_block.isChecked():
            if (self.rect_origin is not None) and (self.rect_end is not None):
                q = QPainter(self.pixmap_tmp)
                q.drawPixmap(self.pixmap_tmp.rect(),self.pixmap_tmp)
                q.drawRect(self.rect_origin[0] , #-self.pixmap_origin[0],
                            self.rect_origin[1], #-self.pixmap_origin[1],
                            self.rect_end[0]-self.rect_origin[0],
                            self.rect_end[1]-self.rect_origin[1])
                self.disp_canvas.setPixmap(self.pixmap_tmp)
        if self.check_brush.isChecked():
            qp = QPainter(self.pixmap_tmp)
            if len(self.drawn_points)>0:
                qpen = QPen(Qt.red)
                qpen.setWidth(self.brush_size)
                qpen.setCapStyle(Qt.RoundCap)
                qp.setPen(qpen)
                qp.drawPixmap(self.pixmap_tmp.rect(),self.pixmap_tmp)
                pts = self.drawn_points[len(self.drawn_points)-1]
                qp.drawPoint(pts[0],pts[1])
            if len(self.erase_points)>0:
                qpen = QPen(Qt.black)
                qpen.setWidth(self.brush_size)
                qpen.setCapStyle(Qt.RoundCap)
                qp.setPen(qpen)
                qp.drawPixmap(self.pixmap_tmp.rect(),self.pixmap_tmp)
                pts = self.erase_points[len(self.erase_points)-1]
                qp.drawPoint(pts[0],pts[1])
            self.disp_canvas.setPixmap(self.pixmap_tmp)
            
    
    def keyPressEvent(self,e):
        '''
        '''
        if self.check_block.isChecked():
            if e.key() == Qt.Key_Enter-1:
                filename = self.image_file_list[self.frame_cnt]
                #self.img = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)
                if self.radio_block_del.isChecked():
                    self.img = apply_modify(self.img,self.rect_origin,self.rect_end,mode="delete")
                if self.radio_block_reserve.isChecked():
                    #self.img = apply_modify(self.img,self.rect_origin,self.rect_end,mode="reserve")
                    self.label = apply_modify(self.label,self.rect_origin,self.rect_end,mode="reserve")
                overalyed_img = overlay_image(self.img,self.label)
                self.show_picture_on_label(overalyed_img,self.disp_canvas)
        
        if self.check_brush.isCheckable():
            if e.key() == Qt.Key_Enter-1:
                filename = self.image_file_list[self.frame_cnt]
                #self.img = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)
                mod_img = draw_points_on_label(self.label,self.drawn_points)
                mod_img = draw_points_on_label(mod_img,self.erase_points,mode='del')
                self.show_picture_on_label(mod_img,self.disp_canvas)
                
    ############# augment tab functions ###########
    def on_btn_original_imgdir(self):
        self.orginal_image_dir = QFileDialog.getExistingDirectory(self,"select dir",'./')
        if self.orginal_image_dir == "":
            self.orginal_image_dir = None
        self.print_info(self.orginal_image_dir,clear=False)
    def on_btn_seg_mask_dir(self):
        self.segemtation_image_dir = QFileDialog.getExistingDirectory(self,"select dir",'./')
        if self.segemtation_image_dir == "":
            self.segemtation_image_dir = None
        self.print_info(self.segemtation_image_dir,clear=False)
    def on_btn_sel_background(self):
        self.aug_background_image_file = QFileDialog.getOpenFileName(self,"openfile","./","*.png *.jpg")[0]
        if self.aug_background_image_file == "":
            self.aug_background_image_file = None
        self.print_info(self.aug_background_image_file,clear=False)
    def on_btn_output_dir(self):
        self.aug_output_folder = QFileDialog.getExistingDirectory(self,"select dir",'./')
        if self.aug_output_folder == "":
            self.aug_output_folder = None
        self.print_info(self.aug_output_folder,clear = False)

    def on_btn_start_augment(self):
        if (self.orginal_image_dir is None) or \
            (self.segemtation_image_dir is None) or \
            (self.aug_background_image_file is None) or \
            (self.aug_output_folder is None):
            self.print_info("Failed",clear=True)
            return
        original_image_files = walk_image_dir(self.orginal_image_dir,"*.png",'*.jpg')
        self.print_info("Totally image {} files loaded".format(len(original_image_files)),clear=True)
        segment_mask_files = walk_image_dir(self.segemtation_image_dir,"*.png","*.jpg")
        self.print_info("Totally mask {} files loaded".format(len(segment_mask_files)))
        if len(original_image_files)!=len(segment_mask_files):
            self.print_info("Numbers not match...",clear=True)
            return
        import cv2
        background = cv2.imread(self.aug_background_image_file,-1)
        for ori_file,mask_file in zip(original_image_files,segment_mask_files):
            ori_img = cv2.imread(ori_file,-1)
            mask = cv2.imread(mask_file,-1)
            newimg = mask_on_new_background(ori_img,mask,background)
            nfname = os.path.join(self.aug_output_folder,os.path.basename(ori_file))
            cv2.imwrite(nfname,newimg)
            self.print_info("writing to {}".format(os.path.basename(ori_file)))


    ############# manual_label tab function #######
    def on_check_enable_labeling(self,state):
        if state: # enabled
            self.progress_bar_ahead.setEnabled(True)
            self.progress_bar_ahead.setMaximum(len(self.image_file_list))
        else:
            self.progress_bar_ahead.setEnabled(False)


    def on_add_new_label_value(self):
        label_name = self.lineEdit_label_name.text().strip()
        label_value = self.lineEdit_label_value.text().strip()
        
        if label_name+label_value == "":
            self.print_info("Empty",clear=True)
            return 
        item_str = "{}:{}".format(label_name,label_value)
        item = QListWidgetItem(item_str)
        for i in range(self.label_name_value_listWidget.count()):
            tmp_item = self.label_name_value_listWidget.item(i)
            tmp_str = tmp_item.text()
            if tmp_str == item_str:
                self.print_info("Cannot add item with same name/value",clear=True)
                return 
        item.setCheckState(False)
        self.label_name_value_listWidget.addItem(item)
        self.print_info("Label added: {}".format(item_str),clear=True)

    def on_confirm_labeling(self):
        lower = min(self.frame_cnt,self.frame_cnt_ahead)
        upper = max(self.frame_cnt,self.frame_cnt_ahead)
        self.print_info("Labeling segment: {} to {}".format(lower,upper),clear=True)
        
        label_value = ""
        for i in range(self.label_name_value_listWidget.count()):
            tmp_item = self.label_name_value_listWidget.item(i)
            if tmp_item.checkState():
                label_value = tmp_item.text().split(":")[-1]
                continue
        if label_value == "":
            self.print_info("Not label value chosen")
            return
        self.labeling_info.append((lower,upper,label_value))


    def on_generate_label_file(self):
        if len(self.labeling_info)<1:
            self.print_info("Nothing to be write",clear=True)
            return
        path = QFileDialog.getExistingDirectory(self,"select dir",'./')
        if path=="":
            self.print_info("Path selected is invalid!",clear=True)
            return 
        all_lines = []
        for ele in self.image_file_list:
            all_lines.append([ele,self.default_label_value])
        for ele in self.labeling_info:
            start,end,value = ele
            for i in range(start,end+1):
                all_lines[i][-1] = value
        filename = os.path.join(path,"label.txt")
        with open(filename,"w+") as fp:
            for ele in all_lines:
                line = "{}#{}\n".format(os.path.basename(ele[0]),ele[1])
                fp.write(line)
            self.print_info("Saving label file to {}/label.txt".format(path),clear=True)

    def rightClickContextMenu(self,pos):
        #print(pos)
        globalPos = self.label_name_value_listWidget.mapToGlobal(pos)
        menu = QMenu()
        item = menu.addAction("Delete")
        item.triggered.connect(self.menuItemClicked)
        menu.exec(globalPos)
        

    def menuItemClicked(self):
        row_id = self.label_name_value_listWidget.currentRow()
        if row_id<0:
            return
        self.label_name_value_listWidget.takeItem(row_id)
        

    ############ RTMP  tab functions ##############
    def on_rtmp_btn(self):
        if self.rtmp_btn.text().find("Play")>-1:
            self.rtmp_btn.setText("Pause RTMP Stream")
            try:
                self.webcam.release()
            except Exception as e:
                self.print_info(e)
            rtmp_link = self.rtmp_link_edit.toPlainText().strip()
            if rtmp_link == "":
                rtmp_link = self.rtmp_link
            self.print_info("connecting link: ["+rtmp_link+"]",clear=True)
            self.webcam = cv2.VideoCapture(rtmp_link)
            #self.webcam.open()
            _,img = self.webcam.read()
            if img is None:
                self.print_info("rtmp link invalid")
                self.rtmp_btn.setText("Play RTMP Stream")
            else:
                # start rtmp timer
                self.rtmp_timer.start(self.refresh_timeout)
        else:
            self.rtmp_btn.setText("Play RTMP Stream")
            self.rtmp_timer.stop()
            try:
                self.webcam.release()
            except Exception as e:
                self.print_info(e)
    def on_save_rtmp2video_btn(self):
        self.print_info("This function has not been implemented...")
    
    def on_rtmp_timer(self):
        _,img = self.webcam.read()
        if img is not None:
            self.show_picture_on_label(img,self.disp_canvas)