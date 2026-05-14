import dataclasses

# Data classes of retrieval / inference policies


@dataclasses.dataclass(frozen=True)
class RetrievalPolicy:
    # Whether to invoke retrieval at all
    run_retrieval: bool = False

    # Retrieval pattern can be either "once" or "interval" (every n tokens)
    retrieval_pattern: str | None = None
    retrieval_interval: int | None = None

    # Query expansion model
    run_query_expansion: bool = False
    query_expansion_seq_len_inference_prefill: int | None = None
    query_expansion_seq_len_prefill_with_template: int | None = (
        None  # prefill = query expansion instruction template + question
    )
    query_expansion_dec_steps: int | None = None

    # Per-user fanout introduced by the query rewriter: one rewriter call
    # produces this many sub-queries, each of which hits ``encode`` and
    # ``retrieval`` independently. ``None``/1 means no fanout (1 rewrite
    # per user request, the historical default). Mirrors how
    # ``passage_reranker_topk`` multiplies workload at the reranker stage.
    query_expansion_fanout: int | None = None

    # Passage reranker model
    run_passage_reranker: bool = False
    passage_reranker_seq_len_inference_prefill: int | None = None
    passage_reranker_topk: int | None = None


@dataclasses.dataclass(frozen=True)
class EncodeDBPolicy:
    # Whether to invoke encode db at all
    run_encode_db: bool = False
    # Chunk length is the seq_len_inference_prefill for the encoder model
    encode_seq_len_inference_prefill: int | None = None
    # number of vectors in the database for encode db
    encode_db_nvec: int | None = None


class PhysicalMapping:
    def __init__(
        self,
        num_retrieval_servers: int = 0,
        num_chips: dict[str, int] = {
            "query_expansion_prefill": 0,
            "query_expansion_decode": 0,
            "encode": 0,
            "passage_reranker": 0,
            "passage_filter": 0,
            "passage_compressor_prefill": 0,
            "passage_compressor_decode": 0,
            "prefill": 0,
            "decode": 0,
            "e2e": 0,
        },
        placement_policy: str = "disaggregated",  # 'collocated' or 'disaggregated'
        collocation_strategy: (
            list[list[str]] | None
        ) = None,  # [['retrieval'], ['prefill'], ['decode']]
    ):
        self.num_retrieval_servers = num_retrieval_servers
        self.num_chips = num_chips
        self.placement_policy = placement_policy
        self.collocation_strategy = collocation_strategy

    # Print each of the attribute
    def __str__(self):
        return (
            f"PhysicalMapping(num_retrieval_servers={self.num_retrieval_servers},"
            + f" num_chips={self.num_chips}, placement_policy={self.placement_policy},"
            + f" collocation_strategy={self.collocation_strategy})"
        )
