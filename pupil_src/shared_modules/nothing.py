import cv2
cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
print(int(cap.get(3)))
out = cv2.VideoWriter('vide1.avi', fourcc, 20.0, (640, 480))
while True:

    ret, frame = cap.read()
    cv2.imshow('3 Channel Window', frame)
    # print(frame.shape)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
