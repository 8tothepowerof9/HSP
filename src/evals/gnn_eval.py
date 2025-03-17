import cv2
import numpy as np
import torch
import mediapipe as mp
import networkx as nx
from torch_geometric.data import Data
from .base import BaseEvaluator
from dataset import *


class GNNEvaluator(BaseEvaluator):
    def __init__(self, model, log, device, save=True):
        super().__init__(model, log, device, save)
        self.mp_hands = mp.solutions.hands

    def __extract_hand_landmarks__(self, image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with self.mp_hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
        ) as hands:
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.append(
                            [lm.x, lm.y, lm.z if hasattr(lm, "z") else 0]
                        )  # (x, y, z)
                return np.array(landmarks), image  # Return landmarks and original image

        return None, image  # No hand detected

    def __image_to_graph__(self, image_path):
        landmarks, image = self.__extract_hand_landmarks__(image_path)

        if landmarks is None:
            print("No hand detected in the image.")
            return None, image

        # Step 1: Normalize (translate & scale)
        landmarks = translate_landmarks(landmarks, 0)  # Center the hand at the wrist
        joint_angles = cal_all_finger_angles(landmarks)  # Compute angles
        landmarks = scale_landmarks(landmarks, max_dist=1)  # Scale

        # Step 2: Construct edges using HAND_CONNECTIONS
        G = nx.Graph()
        for idx, landmark in enumerate(landmarks):
            G.add_node(idx, pos=(landmark[0], landmark[1], landmark[2]))

        for edge in self.mp_hands.HAND_CONNECTIONS:
            G.add_edge(edge[0], edge[1])

        # Convert adjacency matrix to COO format
        adjacency_matrix = nx.adjacency_matrix(G)
        coo = adjacency_matrix.tocoo()
        edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)

        # Step 3: Create node features (x, y, z, angle)
        features = np.hstack([landmarks, np.zeros((landmarks.shape[0], 1))])

        for joint, (_, p2, _) in FINGER_JOINTS.items():
            angle = joint_angles.get(joint, 0)
            features[p2, -1] = angle  # Assign angle to middle joint

        x = torch.tensor(features, dtype=torch.float)

        # Create PyG Data object
        graph_data = Data(x=x, edge_index=edge_index)

        return graph_data, image

    def static_predict(self, image_path):
        self.model.eval()  # Set model to evaluation mode
        graph_data, _ = self.__image_to_graph__(image_path)

        if graph_data is None:
            return "No hand detected! Make sure the tested image contains a hand."

        graph_data = graph_data.to(self.device)

        with torch.no_grad():
            output = self.model(graph_data)
            pred = output.argmax(dim=1).item()  # Get predicted class index

        predicted_label = [key for key, val in LABEL_MAPPING.items() if val == pred][0]
        return predicted_label

    def realtime_predict(self):
        cap = cv2.VideoCapture(0)  # Open webcam (0 for default camera)
        mp_drawing = mp.solutions.drawing_utils  # Utility for drawing
        mp_hands = mp.solutions.hands  # Mediapipe hands model

        with mp_hands.Hands(
            static_image_mode=False,  # Continuous tracking
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                predicted_label = "No hand detected"

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.append(
                                [lm.x, lm.y, lm.z if hasattr(lm, "z") else 0]
                            )

                        landmarks = np.array(landmarks)

                        # Normalize and Scale Landmarks
                        landmarks = translate_landmarks(
                            landmarks, 0
                        )  # Center the hand at the wrist
                        joint_angles = cal_all_finger_angles(landmarks)
                        landmarks = scale_landmarks(landmarks, max_dist=1)

                        # Construct Graph
                        G = nx.Graph()
                        for idx, landmark in enumerate(landmarks):
                            G.add_node(idx, pos=(landmark[0], landmark[1], landmark[2]))

                        for edge in self.mp_hands.HAND_CONNECTIONS:
                            G.add_edge(edge[0], edge[1])

                        adjacency_matrix = nx.adjacency_matrix(G)
                        coo = adjacency_matrix.tocoo()
                        edge_index = torch.tensor(
                            np.vstack((coo.row, coo.col)), dtype=torch.long
                        )

                        # Create Node Features (x, y, z, angle)
                        features = np.hstack(
                            [landmarks, np.zeros((landmarks.shape[0], 1))]
                        )
                        for joint, (_, p2, _) in FINGER_JOINTS.items():
                            angle = joint_angles.get(joint, 0)
                            features[p2, -1] = angle  # Assign angle to middle joint

                        x = torch.tensor(features, dtype=torch.float)
                        graph_data = Data(x=x, edge_index=edge_index)

                        # Model Inference
                        self.model.eval()
                        graph_data = graph_data.to(self.device)

                        with torch.no_grad():
                            output = self.model(graph_data)
                            pred = output.argmax(dim=1).item()

                        predicted_label = [
                            key for key, val in LABEL_MAPPING.items() if val == pred
                        ][0]

                # Display Prediction on the Video Feed
                cv2.putText(
                    frame,
                    f"Predicted: {predicted_label}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # Show the frame
                cv2.imshow("Real-Time Hand Gesture Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
                    break

        cap.release()
        cv2.destroyAllWindows()

    def evaluate(self, dataloader):
        pass

    def plot_history(self):
        pass

    def confusion_matrix(self, dataloader):
        pass
