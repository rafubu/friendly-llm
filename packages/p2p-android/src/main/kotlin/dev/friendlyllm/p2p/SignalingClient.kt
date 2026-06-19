package dev.friendlyllm.p2p

import com.google.gson.Gson
import dev.friendlyllm.p2p.models.IceCandidateMessage
import dev.friendlyllm.p2p.models.OfferMessage
import dev.friendlyllm.p2p.models.RegisterMessage
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import okhttp3.*

class SignalingClient(
    private val signalingUrl: String,
    private val apiKey: String,
    private val scope: CoroutineScope
) {
    private val client = OkHttpClient.Builder()
        .pingInterval(30, java.util.concurrent.TimeUnit.SECONDS)
        .build()
    private var webSocket: WebSocket? = null
    private val gson = Gson()
    private val messageChannel = Channel<String>(Channel.UNLIMITED)

    sealed class SignalingMessage {
        data class Offer(val sdp: String, val roomId: String) : SignalingMessage()
        data class IceCandidate(
            val candidate: String,
            val sdpMid: String?,
            val sdpMLineIndex: Int?,
            val roomId: String
        ) : SignalingMessage()
        data class Error(val message: String) : SignalingMessage()
        object Connected : SignalingMessage()
        object Disconnected : SignalingMessage()
    }

    val incomingMessages: Channel<SignalingMessage> = Channel(Channel.UNLIMITED)

    fun connect() {
        val request = Request.Builder()
            .url(signalingUrl)
            .addHeader("Authorization", "Bearer $apiKey")
            .build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                scope.launch { incomingMessages.send(SignalingMessage.Connected) }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                scope.launch { handleServerMessage(text) }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                scope.launch {
                    incomingMessages.send(SignalingMessage.Disconnected)
                    delay(5000)
                    connect()
                }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                scope.launch {
                    incomingMessages.send(SignalingMessage.Disconnected)
                    delay(5000)
                    connect()
                }
            }
        })
    }

    fun register(modelId: String, vramMb: Int, backend: String) {
        val msg = RegisterMessage(
            model_id = modelId,
            vram_mb = vramMb,
            backend = backend
        )
        webSocket?.send(gson.toJson(msg))
    }

    fun sendOffer(sdp: String, roomId: String) {
        val msg = OfferMessage("offer", sdp, roomId)
        webSocket?.send(gson.toJson(msg))
    }

    fun sendIceCandidate(candidate: String, sdpMid: String?, sdpMLineIndex: Int?, roomId: String) {
        val msg = IceCandidateMessage(
            "ice_candidate", candidate, sdpMid, sdpMLineIndex, roomId
        )
        webSocket?.send(gson.toJson(msg))
    }

    fun disconnect() {
        webSocket?.close(1000, "Node shutting down")
    }

    private suspend fun handleServerMessage(text: String) {
        try {
            @Suppress("UNCHECKED_CAST")
            val json = gson.fromJson(text, Map::class.java) as Map<String, Any>

            when (json["type"]) {
                "offer" -> {
                    incomingMessages.send(SignalingMessage.Offer(
                        sdp = json["sdp"] as String,
                        roomId = json["room_id"] as String
                    ))
                }
                "ice_candidate" -> {
                    incomingMessages.send(SignalingMessage.IceCandidate(
                        candidate = json["candidate"] as String,
                        sdpMid = json["sdp_mid"] as? String,
                        sdpMLineIndex = (json["sdp_m_line_index"] as? Double)?.toInt(),
                        roomId = json["room_id"] as String
                    ))
                }
                "error" -> {
                    incomingMessages.send(SignalingMessage.Error(
                        json["message"] as? String ?: "Unknown error"
                    ))
                }
            }
        } catch (_: Exception) {
            // Ignore malformed messages
        }
    }
}
