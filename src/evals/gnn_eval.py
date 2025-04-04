import cv2
import numpy as np
import torch
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from torch_geometric.data import Data
from .base import BaseEvaluator
from dataset import *
from trainers import REPORT_PATH
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
        orientation_angles = compute_finger_orientation_angles(landmarks)
        landmarks = scale_landmarks(landmarks)  # Scale
        inter_dists = compute_inter_finger_distances(landmarks)

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

        features = np.hstack(
            [
                landmarks,  # (21, 3)
                np.zeros((landmarks.shape[0], 1)),  # angle placeholder
                np.tile(inter_dists, (21, 1)),  # broadcast (21, 10)
                np.tile(orientation_angles, (21, 1)),  # (21, 10)
            ]
        )

        for joint, (_, p2, _) in FINGER_JOINTS.items():
            angle = joint_angles.get(joint, 0)
            features[p2, 3] = angle  # Assign angle to middle joint

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
            output = self.model(graph_data)  # Shape: (1, num_classes)
            probs = torch.exp(output)  # Convert log_softmax output to probabilities
            top_probs, top_indices = torch.topk(probs, k=5, dim=1)

        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()

        # Map indices to labels
        inv_label_map = {v: k for k, v in LABEL_MAPPING.items()}
        top_labels = [inv_label_map[idx] for idx in top_indices]

        # Return list of (label, confidence)
        print(list(zip(top_labels, top_probs)))

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
                        orientation_angles = compute_finger_orientation_angles(
                            landmarks
                        )
                        landmarks = scale_landmarks(landmarks)
                        inter_dists = compute_inter_finger_distances(landmarks)

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
                            [
                                landmarks,
                                np.zeros((landmarks.shape[0], 1)),
                                np.tile(inter_dists, (21, 1)),
                                np.tile(orientation_angles, (21, 1)),
                            ]
                        )
                        for joint, (_, p2, _) in FINGER_JOINTS.items():
                            angle = joint_angles.get(joint, 0)
                            features[p2, 3] = angle  # Assign angle to middle joint

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
        num_batches = len(dataloader)

        # Metrics
        accuracy = self.metrics["accuracy"]
        f1 = self.metrics["f1"]
        accuracy.reset()
        f1.reset()

        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                output = self.model(data)

                # Metrics
                output = output.argmax(dim=1)
                accuracy(output, data.y)
                f1(output, data.y)

        accuracy_score = accuracy.compute().item()
        f1_score = f1.compute().item()

        # Print
        print(
            f"Evaluation Summary: Accuracy: {accuracy_score:.4f} | F1: {f1_score:.4f}"
        )

    def plot_history(self):
        # Load log and plot
        df = pd.read_csv(self.log)
        # Add epoch column
        df["epoch"] = df.index + 1

        # Plot multiple plots
        _, axes = plt.subplots(2, 2, figsize=(15, 10))
        sns.lineplot(x="epoch", y="loss", data=df, ax=axes[0, 0], label="Training Loss")
        sns.lineplot(
            x="epoch", y="val_loss", data=df, ax=axes[0, 0], label="Validation Loss"
        )
        axes[0, 0].set_title("Loss")

        sns.lineplot(
            x="epoch", y="accuracy", data=df, ax=axes[0, 1], label="Training Accuracy"
        )
        sns.lineplot(
            x="epoch",
            y="val_accuracy",
            data=df,
            ax=axes[0, 1],
            label="Validation Accuracy",
        )
        axes[0, 1].set_title("Accuracy")

        sns.lineplot(x="epoch", y="f1", data=df, ax=axes[1, 0], label="Training F1")
        sns.lineplot(
            x="epoch", y="val_f1", data=df, ax=axes[1, 0], label="Validation F1"
        )
        axes[1, 0].set_title("F1 Score")

        plt.tight_layout()
        plt.show()

        if self.save:
            plt.savefig(f"{REPORT_PATH}/{self.model.name}_history.png")

    def confusion_matrix(self, dataloader):
        y_pred = []
        y_true = []

        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                output = self.model(data)

                # Metrics
                output = output.argmax(dim=1)
                y_pred.append(output.cpu().numpy())
                y_true.append(data.y.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABEL_MAPPING.keys(),
            yticklabels=LABEL_MAPPING.keys(),
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
