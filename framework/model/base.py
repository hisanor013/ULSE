class BaseEmbedder:
    """
    Base class for embedding methods.
    Provides common interfaces such as compute_embedding, save_node_embeddings, etc.
    """

    def __init__(self, edge_path, emb_size, output_path, **kwargs):
        self.edge_path = edge_path
        self.emb_size = emb_size
        self.output_path = output_path
        self.embedding = None

    def compute_embedding(self):
        """
        To be implemented in each method.
        """
        raise NotImplementedError(
            "compute_embedding() must be implemented in subclasses"
        )

    def save_node_embeddings(self, path=None):
        """
        Common method to save embedding results to file.
        Can be overridden in subclasses if needed.
        """
        if self.embedding is None:
            raise ValueError("Please run compute_embedding() first")
        if path is None:
            path = self.output_path
        # By default, does nothing. Must be implemented in subclasses.
        raise NotImplementedError(
            "save_node_embeddings() must be implemented in subclasses"
        )

    def get_embedding_statistics(self):

        if self.embedding is None:
            return {"status": "not_computed"}

        nT, d = self.embedding.shape
        return {
            "method": self.__class__.__name__,
            "embedding_shape": self.embedding.shape,
            "embedding_dimension": d,
            "total_embeddings": nT,
        }
