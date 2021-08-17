import cv2

def printBoarderROIS(labelsMatrix,imageToPrint,numOfRows, numOfColumns, widthCells, heightCells):
  for labelsarray in labelsMatrix:
    #print(labelsarray)
      ## Translating the slice coordinates to the original image
    (x1,y1,x2,y2) = labelsarray[0]
    (xb1,yb1,xb2,yb2) = labelsarray[1]
    cv2.rectangle(imageToPrint, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.rectangle(imageToPrint, (xb1, yb1), (xb2, yb2), (255, 0, 0), 2)
  #plt.figure(figsize=(16, 10))
  #plt.imshow(imageToPrint)
  #plt.show()
  return imageToPrint
def printConflictsSolutionROIS(rois,imageToPrint,obj_list):
  for label in rois:
    #print(labelsarray)
      ## Translating the slice coordinates to the original image
    cv2.rectangle(imageToPrint, (label[2], label[3]), (label[4], label[5]), (255, 0, 0), 2)
    cv2.putText(imageToPrint, '{}, {:.3f}'.format(obj_list[label[0]-1], label[1]),
                                (label[2],label[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 1)
  #plt.figure(figsize=(16, 10))
  #plt.imshow(imageToPrint)
  #plt.show()
  return imageToPrint

def drawMatrix(image,numOfRows, numOfColumns, widthCells, heightCells):
  i = 1
  while i<=numOfRows:
    j=1
    while j<=numOfColumns:
      cv2.rectangle(image, ((j*widthCells)-widthCells, (i*heightCells)-heightCells), ((j*widthCells), (i*heightCells)), (0, 0, 255), 2)
      j = j+1
    i = i+1
  return image
def printbboxesoutput(imageToPrint,output):
    obj_list = ['pineapple']
    for bboxCount in range(len(output[0]['rois'])):
            ## Translating the slice coordinates to the original image
            #(x1, y1, x2, y2) = output[0]['rois'][bboxCount].astype(np.int)
            (x1, y1, x2, y2) = output[0]['rois'][bboxCount]
            ## Drawing the bounding boxes and its scores in the original image
            cv2.rectangle(imageToPrint, (x1,y1), (x2,y2), (255, 255, 0), 2)
            obj = obj_list[output[0]['class_ids'][bboxCount]]
            score = float(output[0]['scores'][bboxCount])
            cv2.putText(imageToPrint, '{}, {:.3f}'.format(obj, score),
                                (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 1)
    return imageToPrint
