package dev.friendlyllm.p2p.models

data class RegisterMessage(
    val type: String = "register",
    val model_id: String,
    val vram_mb: Int,
    val backend: String,
    val max_concurrent: Int = 2
)

data class OfferMessage(
    val type: String,
    val sdp: String,
    val room_id: String
)

data class IceCandidateMessage(
    val type: String,
    val candidate: String,
    val sdp_mid: String?,
    val sdp_m_line_index: Int?,
    val room_id: String
)

data class IncomingRequest(
    val model: String,
    val messages: List<IncomingMessage>,
    val stream: Boolean = true,
    val options: Map<String, Any>? = null
)

data class IncomingMessage(
    val role: String,
    val content: String
)

data class InferenceChunk(
    val model: String,
    val response: String,
    val done: Boolean = false,
    val done_reason: String? = null
)
