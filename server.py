from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64
import os
import uvicorn
from starlette.websockets import WebSocketState

app = FastAPI()

# Enable CORS to allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models
def load_model(file_path, label_encoder=False):
    try:
        model_dict = pickle.load(open(file_path, 'rb'))
        model = model_dict['model']
        print(f"Model {file_path} loaded successfully.")
        return (model, model_dict.get('label_encoder')) if label_encoder else model
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None if not label_encoder else (None, None)

model_alphabet = load_model('./model_A_to_Z.p')
model_number = load_model('./model_numbers.p')
model_gujarati, label_encoder_gujarati = load_model('./model_gujarati.p', label_encoder=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9)

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected!")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("message") == "close":
                print("Closing WebSocket by client request...")
                break

            if "frame" in message and "expected_sign" in message and "mode" in message:
                frame_base64 = message["frame"]
                expected_sign = message["expected_sign"]
                mode = message["mode"]

                frame_bytes = base64.b64decode(frame_base64)
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                if frame is not None:
                    data_aux, x_, y_ = [], [], []
                    H, W, _ = frame.shape
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    predicted_character = "?"

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            for lm in hand_landmarks.landmark:
                                x_.append(lm.x)
                                y_.append(lm.y)
                            for lm in hand_landmarks.landmark:
                                data_aux.append(lm.x - min(x_))
                                data_aux.append(lm.y - min(y_))

                        # Prediction
                        try:
                            if mode == "alphabet" and model_alphabet:
                                prediction = model_alphabet.predict([np.asarray(data_aux)])
                                predicted_character = prediction[0].upper()
                            elif mode == "number" and model_number:
                                prediction = model_number.predict([np.asarray(data_aux)])
                                predicted_character = prediction[0].upper()
                            elif mode == "gujarati" and model_gujarati and label_encoder_gujarati:
                                prediction = model_gujarati.predict([np.asarray(data_aux)])
                                predicted_index = prediction[0]
                                predicted_character = label_encoder_gujarati.inverse_transform([predicted_index])[0]
                            print(f"Expected: {expected_sign}, Predicted: {predicted_character}")
                        except Exception as e:
                            predicted_character = "?"
                            print(f"Prediction Error: {e}")

                        # Draw bounding box
                        try:
                            if x_ and y_:
                                x1, y1 = int(min(x_) * W), int(min(y_) * H)
                                x2, y2 = int(max(x_) * W), int(max(y_) * H)
                                cv2.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (0, 0, 0), 2)
                                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Bounding Box Error: {e}")

                    # Encode frame
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    await websocket.send_text(json.dumps({"frame": frame_base64, "prediction": predicted_character}))

    except WebSocketDisconnect:
        print("WebSocket Disconnected by client.")
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:  # Prevent double closing
            print("Closing WebSocket connection...")
            await websocket.close()
        cv2.destroyAllWindows()

# Run the server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))  # Default to 8001 if PORT is not set
    uvicorn.run("server:app", host="0.0.0.0", port=port, ws="websockets", reload=True)
