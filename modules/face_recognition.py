import torch
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
from modules.database import DatabaseManager

class FaceRecognizer:
    """
    Extracts face embeddings using InceptionResnetV1 and compares to known embeddings.
    """
    def __init__(self, device=None, pretrained='vggface2', threshold=0.6, db_path=None):
        # Select device: MPS > CUDA > CPU
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        self.device = device
        # Load pre-trained model
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)
        self.threshold = threshold
        # Optional database manager for whitelist
        self.db = DatabaseManager(db_path) if db_path else None
        # Load known embeddings, ids, and names
        self._load_known()

    def get_embedding(self, face_img):
        """
        Given a BGR face image (numpy array), return 512-d embedding.
        """
        # resize to 160x160
        img = cv2.resize(face_img, (160, 160))
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # to float tensor
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(self.device)
        # normalize to [-1,1]
        tensor = (tensor - 127.5) / 128.0
        # batch dim
        with torch.no_grad():
            emb = self.model(tensor.unsqueeze(0))
        return emb.detach().cpu().numpy()[0]

    @staticmethod
    def cosine_similarity(a, b):
        """
        Compute cosine similarity between two vectors.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def identify(self, emb, known_embeddings, known_ids=None):
        """
        Compare emb (512,) to known_embeddings (N,512).
        Return best match id and similarity or (None, max_sim).
        known_ids: list of identifiers matching known_embeddings.
        """
        if len(known_embeddings) == 0:
            return None, 0.0
        sims = [self.cosine_similarity(emb, ke) for ke in known_embeddings]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        if best_sim >= self.threshold and known_ids:
            return known_ids[best_idx], best_sim
        return None, best_sim

    def _load_known(self):
        """Load embeddings, ids, and names from database."""
        if self.db:
            embeddings, ids, names = self.db.get_embeddings()
            self.known_embeddings = embeddings
            self.known_ids = ids
            self.known_names = names
            self.id_to_name = dict(zip(ids, names))
        else:
            self.known_embeddings = []
            self.known_ids = []
            self.known_names = []
            self.id_to_name = {}

    def recognize_faces(self, face_imgs):
        """
        Recognize list of face images (BGR crops) against known embeddings.
        Returns list of dict: {id, name, similarity, embedding}.
        """
        results = []
        for img in face_imgs:
            # compute embedding
            emb = self.get_embedding(img)
            # identify with DB embeddings
            uid, sim = self.identify(emb, self.known_embeddings, self.known_ids)
            name = self.id_to_name.get(uid)
            results.append({'id': uid, 'name': name, 'similarity': sim, 'embedding': emb})
        return results

