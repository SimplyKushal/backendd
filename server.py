from fastapi import FastAPI, WebSocket
import json
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64

app = FastAPI()

# Load the trained models
try:
    model_dict_alphabet = pickle.load(open('./model_A_to_Z.p', 'rb'))
    model_alphabet = model_dict_alphabet['model']
    print("Alphabet model loaded successfully.")
except Exception as e:
    print(f"Error loading alphabet model: {e}")

try:
    model_dict_number = pickle.load(open('./model_numbers.p', 'rb'))
    model_number = model_dict_number['model']
    print("Number model loaded successfully.")
except Exception as e:
    print(f"Error loading number model: {e}")

try:
    model_dict_gujarati = pickle.load(open('./model_gujarati.p', 'rb'))
    model_gujarati = model_dict_gujarati['model']
    label_encoder_gujarati = model_dict_gujarati['label_encoder']
    print("Gujarati model loaded successfully.")
except Exception as e:
    print(f"Error loading Gujarati model: {e}")

# Gujarati to English phonetic mapping
gujarati_to_english = {
    'ક': 'ka', 'ખ': 'kha', 'ગ': 'ga', 'ઘ': 'gha', 'ઙ': 'nga',
    'ચ': 'cha', 'છ': 'chha', 'જ': 'ja', 'ણ': 'na',
    'ત': 'ta', 'થ': 'tha', 'દ': 'da', 'ધ': 'dha', 'ન': 'na',
    'પ': 'pa', 'ફ': 'pha', 'બ': 'ba', 'ભ': 'bha', 'મ': 'ma',
    'ર': 'ra', 'લ': 'la', 'વ': 'va', 'ળ': 'la', 'શ': 'sha', 'સ': 'sa',
    'હ': 'ha', 'ક્ષ': 'ksha', 'જ્ઞ': 'jna', 'ટ': 'ta', 'ઠ': 'tha',
    'ડ': 'da', 'ઢ': 'dha'
}
english_to_gujarati = {v: k for k, v in gujarati_to_english.items()}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9)

@app.websocket("/ws")
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
                            if mode == "alphabet":
                                prediction = model_alphabet.predict([np.asarray(data_aux)])
                                predicted_character = prediction[0].upper()
                            elif mode == "number":
                                prediction = model_number.predict([np.asarray(data_aux)])
                                predicted_character = prediction[0].upper()
                            elif mode == "gujarati":
                                prediction = model_gujarati.predict([np.asarray(data_aux)])
                                predicted_index = prediction[0]
                                predicted_character = label_encoder_gujarati.inverse_transform([predicted_index])[0]
                                predicted_character = english_to_gujarati.get(predicted_character, predicted_character)
                            print(f"Expected: {expected_sign}, Predicted: {predicted_character}")
                        except Exception as e:
                            predicted_character = "?"
                            print(f"Prediction Error: {e}")

                        # Draw bounding box
                        try:
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
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        print("Closing WebSocket connection...")
        await websocket.close()
        cv2.destroyAllWindows()

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, ws="websockets", reload=True)