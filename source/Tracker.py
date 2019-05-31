# -*- coding: utf-8 -*-

import cv2 as cv
from collections import OrderedDict

from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment

from source.Kalman import KF
from source.fast_nms import fast_non_max_suppression
import numpy as np
import time

"""
    Gerencia objetos da classe Tracker e derivados   
"""
class MultiTracker:

    def __init__(self, tracker_algorithm = 'Kalman', assingment_algorithm = 'IoU'):

        # Lista de objetos tracker
        self.tracker_algorithm = tracker_algorithm
        self._tracker_list = OrderedDict()
        
        # Valores de retorno dos trackers
        self._bbox_list     = OrderedDict()
        self._ok_list       = OrderedDict()
        self._time_list     = OrderedDict()
        self._last_time_list= OrderedDict()
        self._last_pos_list = OrderedDict()
        
        # Referencia de quando apagar um tracer
        self._unseen_list  = OrderedDict()

        # Lista de id dos trackers
        self._id_list       = list()
        self._last_id       = -1

        self.thresh         = 0.2
        self.prop           = 0.01
        self._max_time      = 100

        # Out of border termination threhsold condition
        self.min_dist_from_border = 100 

        # Lista de eventos
        self.eventlist      = []

        # Variaveis relevantes
        self.attendment_time_thresh = 10 
        self._max_unseen = 150

    """
    
    """
    def add_tracker(self, frame, bbox):

        # try:

        # tracker_obj = cv.TrackerMedianFlow_create()
        if self.tracker_algorithm == 'CSRT':
            tracker_obj = cv.TrackerCSRT_create()
            ok = tracker_obj.init(frame, tuple(bbox))
        elif self.tracker_algorithm == 'Kalman':
            tracker_obj = KF()
            ok = tracker_obj.init(0.0, bbox)

        # tracker_obj = cv.TrackerTLD_create()
        # Inicializa o objeto

        # Prepara a insersao nas estruturas
        self._last_id += 1
        self._id_list.append(self._last_id)

        # Insere info nas estruturas
        self._tracker_list[self._last_id] = tracker_obj
        self._bbox_list[self._last_id] = bbox
        self._ok_list[self._last_id] = ok
        self._time_list[self._last_id] = time.time()
        self._unseen_list[self._last_id] = 0       
        self._last_time_list[self._last_id]= time.time()
        self._last_pos_list[self._last_id] = bbox[0]
        # except:
            # print ('error')

    def realocate_old(self, new_id, old_id):

        if old_id in self._id_list and new_id in self._id_list:

            # pop(key) para os dicionarios
            self._tracker_list[old_id] = self._tracker_list[new_id]
            self._bbox_list[old_id] = self._bbox_list[new_id]
            self._ok_list[old_id] = self._ok_list[new_id]
            self._unseen_list[old_id] = self._unseen_list[new_id]
            self._time_list[old_id] = self._time_list[new_id]
            self._last_time_list[old_id] = self._last_time_list[new_id]
            self._last_pos_list[old_id] = self._last_pos_list[new_id]

            # pop(key) para os dicionarios
            self._tracker_list.pop(new_id)
            self._bbox_list.pop(new_id)
            self._ok_list.pop(new_id)
            self._unseen_list.pop(new_id)
            self._time_list.pop(new_id)
            self._last_time_list.pop(new_id)
            self._last_pos_list.pop(new_id)

            if new_id in self._id_list:
                # remove(value) para a lista
                self._id_list.remove(new_id)


    def remove_by_id(self, id_to_remove, merge = False, renew = False, update = False):

        # Verifica se o id existe na lista de ids ativos
        if id_to_remove in self._id_list:

            if time.time() - self._time_list[id_to_remove] > self.attendment_time_thresh and merge == False:
                print ('{} Finalizado: {}'.format(id_to_remove, time.time() - self._time_list[id_to_remove]))
                # self.eventlist.append( ('atendimento concluido' , False) )

            # Remove todas as entradas correspondentes ao id
            
            # pop(key) para os dicionarios
            self._tracker_list.pop(id_to_remove)
            self._bbox_list.pop(id_to_remove)
            self._ok_list.pop(id_to_remove)
            self._unseen_list.pop(id_to_remove)
            self._time_list.pop(id_to_remove)
            self._last_time_list.pop(id_to_remove)
            self._last_pos_list.pop(id_to_remove)
        
            # remove(value) para a lista
            self._id_list.remove(id_to_remove)

            scope = 'Merge' if merge else 'Renew' if renew else 'Updade'
            print ("{} Removed at {}".format(id_to_remove, scope))

    """
        Retorna lista de bounding boxes
    """
    def get_objects(self):

        return_list = list()
        keys = []
        time = []
        for key in self._bbox_list.keys():
            return_list.append(self._bbox_list[key])
            keys.append(key)
            time.append(self._time_list[key])

        return return_list, keys, time, self._bbox_list, self._bbox_list.keys()
    """
        Retorna informaÃ§Ã£o para uso interno
    """
    def get_internal(self):
        _last_time_ret = list()
        _last_pos_ret = list()
        for key in self._bbox_list.keys():
            _last_time_ret.append(self._last_time_list[key])
            _last_pos_ret.append(self._last_post_list[key])

        return _last_time_ret, _last_pos_ret
    """ 
        Retorna a bounding box do ultimo objeto inserido
    """
    def get_last(self):

        if self._id_list[-1] > 0:
            return self._bbox_list[self._id_list[-1]]

        else:
            return None

    """
        Retorna o centro da bounding box
    """
    def get_bbox_center(self, box):

        return ( int((2*box[0]+box[2])/2), int((2*box[1]+box[3])/2))

    def bb_intersection_over_union(self, boxA, boxB):

        boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
        boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou
    
    def bb_intersection_prop(self, boxA, boxB):

        bA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
        bB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(bA[0], bB[0])
        yA = max(bA[1], bB[1])
        xB = min(bA[2], bB[2])
        yB = min(bA[3], bB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        bAArea = (bA[2] - bA[0] + 1) * (bA[3] - bA[1] + 1)
        bBArea = (bB[2] - bB[0] + 1) * (bB[3] - bB[1] + 1)
        bArea = bBArea
        box = boxA if bAArea < bBArea else boxB
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(bArea)

        # return the intersection over union value
        return iou, box

    '''
        Metodo para decidir se o merge deve acontecer
    '''
    def get_merge_matrix(self, bbox_list):
        num_new_boxes = len(bbox_list)
        num_old_boxes = len(bbox_list)
        # Matriz com:
        #    len(bbox_list) colunas e kinhas
        iou_matrix = np.ndarray((num_old_boxes, num_new_boxes), dtype=np.float32)
        # Preenche a matrix de distancia
        for j, new_box in enumerate(bbox_list):
            for i, old_box in enumerate(bbox_list):
                iou_matrix[i][j], box = self.bb_intersection_prop(old_box, new_box)
                # print (iou_matrix[i][j])
    
        return iou_matrix
    '''
        Metodo para medir os IoU minimos
    '''
    def get_IoU_matrix(self, new_bbox_list, old_bbox_list):
        
        num_new_boxes = len(new_bbox_list)
        num_old_boxes = len(old_bbox_list)
        # Matriz com:
        #    len(old_bbox_list) linhas e
        #    len(new_bbox_list) colunas 
        iou_matrix = np.ndarray((num_old_boxes, num_new_boxes), dtype=np.float32)
        # Preenche a matrix de distancia
        for j, new_box in enumerate(new_bbox_list):
            for i, old_box in enumerate(old_bbox_list):
                iou_matrix[i][j] = self.bb_intersection_over_union(old_box, new_box)
                # print (iou_matrix[i][j])
    
        return iou_matrix
    '''
        Metodo para pegar a matriz de distancias
    '''
    def get_distance_matrix(self, new_bbox_list, old_bbox_list):

        num_new_boxes = len(new_bbox_list)
        num_old_boxes = len(old_bbox_list)
        # Matriz com:
        #    len(old_bbox_list) linhas e
        #    len(new_bbox_list) colunas 
        dist_matrix = np.ndarray((num_old_boxes, num_new_boxes), dtype=np.float32)
        # Preenche a matrix de distancia
        for j, new_box in enumerate(new_bbox_list):
            center_new = (new_box[0], new_box[1]) # self.get_bbox_center(new_box)

            for i, old_box in enumerate(old_bbox_list):
                center_old = (old_box[0], old_box[1]) # self.get_bbox_center(old_box)
                dist_matrix[i][j] = euclidean(center_new, center_old)
    
        return dist_matrix
    '''
        Metodo para pegar os menores matches validos da matriz de distancias
    '''
    def get_minimal_matches(self, dist_matrix, thresh, op = True):
        
        # Para cada nova bbox, acha a melhor antiga e salva o indice
        new_to_old_matches  = []

        # Ordena a matriz de distancia pelos indices que apresentam menor distancia        
        dist_flat = dist_matrix.flatten(0)
        sorting = np.argsort(dist_flat)[::-1]
        repeated = []
        used_new = []

        for i in sorting:
            # try:
            # Acha os indices da matriz original baseado na matriz flat
            o = int(i / dist_matrix.shape[1])
            n = i % dist_matrix.shape[1]
            # Se a distancia for menor que o threshold
            if dist_matrix[o][n] > thresh:
                # Adiciona os pares de indice na lista de achados
                if op:
                    if o not in repeated and n not in used_new:
                        new_to_old_matches.append((o, n))
                        repeated.append(o)
                        used_new.append(n)
                else:
                    new_to_old_matches.append((o, n))
                    repeated.append(o)
                    used_new.append(n)

        return new_to_old_matches, used_new


    """
        Metodo para atualizar a posicao dos objetos com a previsao do classificador.
        
        Nao existe associacao explicita entre as bounding boxes fornecidas pelo classificador e os
        objetos previamente rastreados pelos trackers.

        O relacionamento entre as novas boxes e as antigas sera dado pela distancia euclidiana,
        ou seja, para cada nova bounding box, a box antiga mais proxima e substituida.

        Caso uma box antiga fique sem relacao com nenhuma nova, o tracker correspondente sera removido
        da lista de trackers.
    """
    def feed_tracker(self, frame, new_bbox_list, dt, unseen_check=False):

        if new_bbox_list != []:
            old_bbox_list, keys, _, _, _ = self.get_objects()

            num_new_boxes = len(new_bbox_list)
            num_old_boxes = len(old_bbox_list)

            if not old_bbox_list or old_bbox_list == ([],[]):
                for new in new_bbox_list:
                    self.add_tracker(frame, new)     
                    return True  

            iou_matrix = self.get_IoU_matrix(new_bbox_list, old_bbox_list)
            # Para cada nova bbox, acha a melhor antiga e salva o indice
            new_to_old_matches, used_new = self.get_minimal_matches(iou_matrix, self.thresh)

            # atualiza
            for o, n in new_to_old_matches:
                
                if keys[o] in self._tracker_list.keys():

                    new_bbox_list[n] = (new_bbox_list[n][0],
                                        new_bbox_list[n][1],
                                        new_bbox_list[n][2],
                                        new_bbox_list[n][3])

                    
                        
                    # !! FIXME !! Is this "or" condition correct?
                    if (self._unseen_list[keys[o]] > 1) or (iou_matrix[o][n] < 0.9):
                        
                        if self.tracker_algorithm == 'CSRT':
                            print("CORRIGIU TRACKER {}".format(keys[o]))
                            self._tracker_list[keys[o]] = cv.TrackerCSRT_create()
                            self._tracker_list[keys[o]].init(frame, new_bbox_list[n])
                            
                
                        elif self.tracker_algorithm == 'Kalman':
                            self._bbox_list[keys[o]] = self._tracker_list[keys[o]].update(dt, new_bbox_list[n])
                        
                        if (unseen_check is True) and (self._unseen_list[keys[o]] > 0):
                            self._unseen_list[keys[o]] -= 1
                else:
                    raise Exception("Key error traversing tracked objects")

            used_old = [o for (o,n) in new_to_old_matches]

            #  Se tiver mais novas do que antigas, tem que adicionar
            if num_new_boxes - len(used_new) > 0:
                unused_new = [n for n in range(num_new_boxes) if n not in used_new]

                for index in unused_new:
                    self.add_tracker(frame, new_bbox_list[index])

            # Senao, tem que apagar as que sobraram
            unused_old = [o for o in range(num_old_boxes) if o not in used_old]
            used_old = [o for (o, _) in new_to_old_matches]
            
            if unused_old != []:
                for index in unused_old:
                    self._unseen_list[keys[index]] += 1
                    if (time.time() - self._last_time_list[keys[index]] > self._max_time):
                        self.remove_by_id(keys[index], renew = True)
        else:
            old_bbox_list, keys, _, _, _ = self.get_objects()
            for key in keys:
                self._unseen_list[key] += 1
            
        self.update(frame, dt)
        


    def update(self, image, dt):
        """
        Alimenta os trackers com o novo frame. Atualiza-os, obtendo uma nova bounding box.
        Caso o tracker nao encontre o objeto, incrementa sua entrada correspondente em _unseen_list.
        Trackers que possuem valor _useen_list acima de _max_unseen sao removidos de todas as estruturas.
        """
        if self.tracker_algorithm == 'CSRT':
            try:
                for _id in self._id_list:
                    _ok, _bbox = self._tracker_list[_id].update(image)
                    self._ok_list[_id] = _ok

                    if (_ok is True):
                        self._bbox_list[_id] = _bbox

                    if (not _ok):
                        self._unseen_list[_id] += 1    
                        # Need to check if the tracked object is too close from frame borders
                        # out_of_boundary_x = (_bbox[0] < self.min_dist_from_border) or (_bbox[2] > (image.shape[1] - self.min_dist_from_border))
                        # out_of_boundary_y = (_bbox[1] < self.min_dist_from_border) or (_bbox[3] > (image.shape[0] - self.min_dist_from_border)) 
                    
                        # out_of_boundary = out_of_boundary_x or out_of_boundary_y
                        # if out_of_boundary:
                        #     self.remove_by_id(_id, update = True)

                    if self._unseen_list[_id] > self._max_unseen:
                        self.remove_by_id(_id, update = True)
                    
                    if not _bbox == self._last_pos_list[_id]:
                        self._last_time_list[_id] = time.time()
                    
                    self._last_pos_list[_id] = _bbox
        
            except KeyError:
                print ('key error')
        
        if self.tracker_algorithm == 'Kalman':
            try:
                
                for _id in self._id_list:
                    print("unseen_list[id:{}] = {}".format(_id, self._unseen_list[_id]))
                    if self._unseen_list[_id] > self._max_unseen:
                        self.remove_by_id(_id, update=True)
                        
            except KeyError:
                print ('key error')       



    def nms_match (self, boxes, keys, overlap_threshold):

        if len(boxes) <= 1:
            return boxes, keys

        boxes = [(float(box[0]), float(box[1]), float(box[2]), float(box[3])) for box in boxes]

        pick = []

        x1, y1, x2, y2 = zip(*[(x1,y1,x2,y2) for x1,y1,x2,y2 in boxes])
        x1, y1, x2, y2 = list(x1), list(y1), list(x2), list(y2) 

        unzip_this = [((x2 - x1 + 1) + (y2 - y1 + 1), key) for x1, y1, x2, y2, key in zip(x1, y1, x2, y2, keys)]

        area, kys = zip(*unzip_this)
 
        idxs = np.argsort(y2)

        while len(idxs) > 0:

            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], [x1[index] for index in idxs[:last]] if idxs[:last] != [] else x1[0])
            yy1 = np.maximum(y1[i], [y1[index] for index in idxs[:last]] if idxs[:last] != [] else y1[0])
            xx2 = np.minimum(x2[i], [x2[index] for index in idxs[:last]] if idxs[:last] != [] else x2[0])
            yy2 = np.minimum(y2[i], [y2[index] for index in idxs[:last]] if idxs[:last] != [] else y2[0])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / [area[index] for index in idxs[:last]] if idxs[:last] != [] else area[0]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

        return [box for index, box in enumerate(boxes) if index in pick], [key for index, key in enumerate(kys) if index in pick]

    def merge(self, frame):

        bbox_list, keys, _, _, _ = self.get_objects()
        # inter_prop_matrix = self.get_merge_matrix(bbox_list)
        boxes, kys = self.nms_match(bbox_list, keys, 10)
        for index, key in enumerate(kys):
            self._bbox_list[key] = boxes[index]
        to_del = [key for key in keys if (key not in kys)]
        print (kys, keys)
        for key in to_del:
            self.remove_by_id(key, merge=True)