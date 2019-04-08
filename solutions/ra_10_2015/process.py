import time
import numpy as np
import cv2
import math
from keras.models import load_model
from solution.ra_10_2015.Check import is_intersect


def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized


def distance(point1,point2):
    return math.sqrt(math.pow(point2[1] - point1[1], 2) + math.pow(point2[0] - point1[0], 2))


def find_lines(color_img):
    green_frame = color_img[:, :, 1]
    blue_frame = color_img[:, :, 0]

    #cv2.imshow('Window', green_frame)
    #cv2.waitKey(1)
    #time.sleep(5)

    kernel = np.ones((3, 3), np.uint8)
    blue_frame = cv2.erode(blue_frame, kernel, iterations=1)
    blue_frame = cv2.dilate(blue_frame, kernel, iterations=1)
    green_frame = cv2.erode(green_frame, kernel, iterations=1)
    max_line_gap = 86
    min_line_length = 201
    p_blue = cv2.Canny(blue_frame, threshold1=202, threshold2=298)
    p_green = cv2.Canny(green_frame, threshold1=202, threshold2=298)
    p_blue = cv2.GaussianBlur(p_blue, (7, 7), 0)
    p_green = cv2.GaussianBlur(p_green, (7, 7), 0)

    lines = cv2.HoughLinesP(p_blue, 1, np.pi / 180, 100, min_line_length, max_line_gap)
    for x1, y1, x2, y2 in lines[0]:
        if distance([x1, y1], [x2, y2]) < 240:
            if len(lines) > 1:
                for j in range(1, len(lines)-1):
                    for x11, y11, x22, y22 in lines[j]:
                        if distance([x11, y11], [x2, y2]) > distance([x1, y1], [x2, y2]):
                            x1 = x11
                            y1 = y11
                        if distance([x1, y1], [x22, y22]) > distance([x1, y1], [x2, y2]):
                            x2 = x22
                            y2 = y22
        cv2.line(color_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
    lines2 = cv2.HoughLinesP(p_green, 1, np.pi / 180, 100, min_line_length, max_line_gap)
    for x1, y1, x2, y2 in lines2[0]:
        cv2.line(color_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
    if showLinesFound:
        cv2.imshow('Lines', color_img)
        cv2.waitKey(1)
    return lines[0][0], lines2[0][0]

    jump = jumpConst
    for frames in images_of_video:
        if jumpConst != 0 and jump < jumpConst:
            jump = jump + 1
            continue
        jump = 0

def loadvideos(video):
    vidcap = cv2.VideoCapture(pathToVideos + video)
    success, image = vidcap.read()
    success = True
    images = []
    while success:
        images.append(image)
        success,image = vidcap.read()
    return images


def select_roi(image_orig):
    red = image_orig[:, :, 2]
    retq, binimage = cv2.threshold(red, 120, 255, 1)
    #cv2.imshow('Window', binimage)
    #cv2.waitKey(0)
    #time.sleep(0.25)
    img, contours, hierarchy = cv2.findContours(binimage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    region_locations = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (50 < area < 360 and (h >= 14 and w > 4)) or (h > 18 and w <= 4 and area > 25): #if h >= 12 and w > 6 and 50 < area < 500:
            modx = x + round(w / 2) - 14
            mody = y + round(h / 2) - 14
            if modx < 0:
                modx = 0
            if mody < 0:
                mody = 0
            region = binimage[mody:mody + 28, modx:modx + 28]
            checker = resize_region(region)
            num, chance = check_with_cnn(checker);
            if chance>0.81:
                region_locations.append([modx, mody])
                regions_array.append([resize_region(region), (modx, mody, 28, 28)])
                cv2.rectangle(image_orig, (modx, mody), (modx+28, mody+28), (0, 255, 0), 2)
    filtered_regions = [region[0] for region in regions_array]
    return image_orig, filtered_regions, region_locations


def check_with_cnn(image):
    number_img = 255 - image
    number_img = number_img/255
    found = sum(sum(number_img))
    if 58 < found < 94:
        number_img = cv2.dilate(number_img, np.ones((2, 2), np.uint8), iterations=1)
    elif found < 58:
        number_img = cv2.dilate(number_img, np.ones((3, 3), np.uint8), iterations=1)

    send = number_img.reshape(1, 28, 28, 1)
    chance = model.predict(send)
    chance = chance[0]
    pop = max(chance)
    number = np.argmax(chance)

    if prikazSlanjaCNN:
        cv2.resizeWindow('CNN', 300, 300)
        cv2.imshow('CNN', image)
        cv2.setWindowTitle('CNN', str(number) + " sa " + str(pop))
        cv2.waitKey(1)
        time.sleep(0.5)
    return number, max(chance)


def normalize(speed):
    length = math.sqrt(math.pow(speed[0], 2) + math.pow(speed[1], 2))
    return [speed[0]/length,speed[1]/length]


def check_position(positionOld, positionNew, speed, missing):
    fixing = jumpConst
    if jumpConst == 0:
        fixing = 1
    elif jumpConst == 1:
        fixing =2
    speeder = [0, 0]
    if speed[0] == 0 and speed[1] == 0:
        speeder[0] = 3
        speeder[1] = 3
    else:
        nor = normalize(speed)
        speeder[0] = (nor[0])
        speeder[1] = (nor[1])
    if positionNew[0] >= positionOld[0] and positionNew[1] >= positionOld[1]:
        for i in range(1, (missing+7)*fixing):
            check_old_position = [positionOld[0] + speeder[0]*i, positionOld[1]+speeder[1]*i]
            dist = distance(positionNew, check_old_position)
            if dist < 42:
                return 1,dist
        return 0,0
    else:
        return 0,0


def getNumberFoundFromList(numberFind,list_found,list_position):
    found = []
    for it in range(0, len(list_found)):
        number = list_found[it]
        position = list_position[it]
        if(numberFind == number):
            found.append([numberFind,position])
    return found

def processFoundNumbersToHistory(got_numbers, got_locations, frame_count):
    for i in range(0, 10):
        array_for_number = []
        if i in dict_numbers:
            array_for_number = dict_numbers.get(i, 'none')
            found_in_frame = getNumberFoundFromList(i, got_numbers, got_locations)
            if len(found_in_frame) > 0:
                for combo_iter in range(0, len(array_for_number)):
                    num_in_dict = array_for_number[combo_iter]
                    last_element = len(num_in_dict[1]) - 1
                    history_position = num_in_dict[1]
                    position_old = history_position[last_element]
                    missing = frame_count - num_in_dict[3]
                    found = -1
                    for find_iter in range(len(found_in_frame)): # Changes 12:10AM 4/6/2019
                        is_it, distance_found = check_position(position_old,found_in_frame[find_iter][1], num_in_dict[2], missing)
                        if is_it == 1:
                            found = find_iter
                            break
                    if found >= 0:
                        array_for_number[combo_iter][2] = [abs(found_in_frame[found][1][0] - history_position[0][0]),abs(found_in_frame[found][1][1] - history_position[0][1])]
                        array_for_number[combo_iter][3] = frame_count
                        history_position.append(found_in_frame[found][1])
                        array_for_number[combo_iter][1] = history_position
                        del found_in_frame[found]
            if len(found_in_frame)>0:
                for new in found_in_frame:
                    array_for_number.append([new[0], [new[1]], [0, 0], frame_count])
            dict_numbers[i] = array_for_number
        else:
            found_new = getNumberFoundFromList(i,got_numbers,got_locations)
            for new in found_new:
                array_for_number.append([new[0], [new[1]], [0, 0], frame_count])
            if len(found_new) > 0:
                dict_numbers[i] = array_for_number


def processFrame(image,frame_count):
    img_locations_selected, regions, locations = select_roi(image.copy())
    got_numbers_locations = []
    chances = []
    got_locations = []
    for iterator in range(0, len(regions)): # regions:
            got_number,chance = check_with_cnn(regions[iterator])
            if chance>0.7:
                got_numbers_locations.append([got_number,locations[iterator],chance])
                # cv2.imwrite("images/4_" + str(random.randint(0, 1000)) +".jpg", regions[iterator])
    if showFoundNumbersWithHold:
        cv2.imshow('Image with numbers', img_locations_selected)
        cv2.setWindowTitle('Image with numbers', 'NumbersFound:' + str(len(regions)))
        cv2.waitKey(1)
        time.sleep(2)
    processFoundNumbersToHistory([item[0] for item in got_numbers_locations], [item[1] for item in got_numbers_locations], frame_count)


def backTracking(positions,blue_line,green_line):
    if len(positions) <= 4:
        return False, False
    fixing = jumpConst
    colide_with_blue = False
    colide_with_green = False
    vec_adj = 1;
    for locations in range(0,1):
        vector_moving = [positions[locations+vec_adj][0] - positions[locations][0], positions[locations+vec_adj][1] - positions[locations][1]]
        for i in range(1, fixing):
            check_positions_to_interesect = [positions[locations][0],positions[locations][1],positions[locations][0] - vector_moving[0] * i, positions[locations][1] - vector_moving[1] * i]
            contact_blueUL = is_intersect(blue_line, check_positions_to_interesect)
            contact_greenUL = is_intersect(green_line, check_positions_to_interesect)
            if contact_blueUL:
                colide_with_blue = True
            if contact_greenUL:
                colide_with_green = True
    return colide_with_blue, colide_with_green

def check_will_it_hit(positions,blue_line,green_line):
    colide_with_blue = False
    colide_with_green = False
    if len(positions) <= 2:
        return False, False
    adding = 4
    for loc in range(0, len(positions)-adding):
        contact_blueUL = is_intersect(blue_line, [positions[loc][0], positions[loc][1], positions[loc+adding][0], positions[loc+adding][1]])
        contact_greenUL = is_intersect(green_line, [positions[loc][0], positions[loc][1], positions[loc+adding][0], positions[loc+adding][1]])
        if contact_blueUL:
            colide_with_blue = True
        if contact_greenUL:
            colide_with_green = True

    return colide_with_blue, colide_with_green

pathToVideos = "../../data/videos/"

model = load_model('model/7od22.h5')
videos = ['video-0.avi', 'video-1.avi', 'video-2.avi', 'video-3.avi', 'video-4.avi', 'video-5.avi', 'video-6.avi', 'video-7.avi', 'video-8.avi', 'video-9.avi']
prikazSlanjaCNN = False
showLinesFound = False
showFoundNumbersWithHold = False
showPomerajPrint = False
slowDebugingShowImageInput = False


array_blue_line = []
array_green_line = []
images_matrix = []
array_dict_numbers = []
jumpConst = 3

print('Start computing')

for processVideo in range(len(videos)):
    images_of_video = (loadvideos(videos[processVideo]))
    dict_numbers = dict()
    first = 1
    counter = 0
    jump = jumpConst
    for frames in images_of_video:
        if jumpConst != 0 and jump < jumpConst:
            jump = jump + 1
            continue
        jump = 0
        if first == 1:
            blue_lineTmp, green_lineTmp = find_lines(images_of_video[0])
            array_blue_line.append(blue_lineTmp)
            array_green_line.append(green_lineTmp)
        if slowDebugingShowImageInput:
            cv2.imshow("ImageProcessing", frames)
            cv2.waitKey(1)
        processFrame(frames,counter)
        first = first + 1
        #if (first % 100) == 0:
        #    print('Done frame=' + str(first))
        counter = counter +1
    print("Done video")
    array_dict_numbers.append(dict_numbers)
    del dict_numbers

file = open("out.txt","w")
file.write("RA 10/2013 Andrija Cvejic\r")
file.write("file\tsum\r")

for j in range(0, len(videos)):
    sum_cal = 0
    HisCheck = 0
    HistoryOperations = ""
    for i in range(0, 10):
        dict_for_video = array_dict_numbers[j];
        array_for_number = dict_for_video.get(i, 'none')  # dict_numbers.get(i, 'none')
        if array_for_number != 'none':
            for combo_iter in range(0, len(array_for_number)):
                history_position = array_for_number[combo_iter][1]
                if len(history_position) > 2:
                    HisCheck = HisCheck + 1
                    blue,green = check_will_it_hit(history_position, array_blue_line[j], array_green_line[j])
                    blue_back = False
                    green_back = False
                    if green or green_back:
                        HistoryOperations = HistoryOperations + "-" + str(array_for_number[combo_iter][0]) + " "
                        sum_cal = sum_cal - array_for_number[combo_iter][0]
                    if blue or blue_back:
                        sum_cal = sum_cal + array_for_number[combo_iter][0]
                        HistoryOperations = HistoryOperations + "+" + str(array_for_number[combo_iter][0]) + " "
    file.write(str(videos[j]) + "\t" + str(sum_cal) + "\r");
    print("Za " + str(videos[j]) + " gde suma=" + str(sum_cal))
file.close()

import solution.ra_10_2015.test as test
test.tacnost()