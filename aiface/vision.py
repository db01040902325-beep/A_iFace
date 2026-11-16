"""Camera-only preview. Bounding boxes are NOT metric head positions or identities."""

def preview_camera(device=0):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('Install opencv-python for camera preview') from exc
    face=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    if face.empty(): raise RuntimeError('Face model not found')
    cam=cv2.VideoCapture(device)
    if not cam.isOpened(): raise RuntimeError('Cannot open camera')
    print('Face preview only: no motor output, no real distance estimate. Press q to close.')
    try:
        while True:
            ok,frame=cam.read()
            if not ok: break
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(55,55))
            for x,y,w,h in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,'FACE - NO DISTANCE',(x,max(15,y-8)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            cv2.imshow('A_iFace camera preview (simulation only)',frame)
            if cv2.waitKey(1)&0xFF==ord('q'):break
    finally:
        cam.release();cv2.destroyAllWindows()
