from mpi4py import MPI
import cv2
import os
import time
from ultralytics import YOLO
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
SAVE_FOLDER = "captures"
MODEL_FILE = "yolov8n.pt" 
def ensure_folder():
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
def clear_folder():
    if not os.path.exists(SAVE_FOLDER):
        return
    for f in os.listdir(SAVE_FOLDER):
        os.remove(os.path.join(SAVE_FOLDER, f))
    print("[MASTER] capture folder cleared")
def save_image(img, index):
    ensure_folder()
    name = f"{SAVE_FOLDER}/capture_{index}.jpg"
    cv2.imwrite(name, img)
    print(f"[MASTER] saved -> {name}")
def worker_loop():
    """
    Worker nodes wait for frames,
    run detection, send results back.
    """
    print(f"[Worker {rank}] loading model...")
    model = YOLO(MODEL_FILE)
    while True:
        frame = comm.recv(source=0)
        if frame is None:
            break
        start = time.time()
        result = model(frame, verbose=False)
        processed = result[0].plot()
        elapsed = time.time() - start
        print(f"[Worker {rank}] inference {elapsed:.3f}s")
        comm.send(processed, dest=0)
def master_loop():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera not detected.")
        return
    print("\nControls: q=quit | s=save | c=clear\n")
    worker_turn = 1
    save_count = 0
    last_time = time.time()
    while True:
        ok, frame = cam.read()
        if not ok:
            print("Frame read failed")
            break
        worker_turn = (worker_turn % (size - 1)) + 1
        comm.send(frame, dest=worker_turn)
        output = comm.recv(source=worker_turn)
        now = time.time()
        fps = 1 / (now - last_time)
        last_time = now
        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.imshow("AirScript HPC Detection", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[MASTER] quitting...")
            break
        elif key == ord('s'):
            save_image(output, save_count)
            save_count += 1
        elif key == ord('c'):
            clear_folder()
    for w in range(1, size):
        comm.send(None, dest=w)
    cam.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    print(f"Process {rank} started (total={size})")
    if size < 2:
        print("Run using mpiexec with at least 2 processes.")
        exit()
    if rank == 0:
        master_loop()
    else:
        worker_loop()
