package dev.friendlyllm.p2p

import dev.friendlyllm.p2p.models.IncomingRequest
import kotlinx.coroutines.*

class P2PConnectorEngine(
    private val signalingUrl: String,
    private val apiKey: String,
    private val engine: InferenceEngine,
    private val modelId: String,
    private val vramMb: Int = 4096,
    private val backend: String = "gpu"
) {
    private var scope: CoroutineScope? = null
    private lateinit var signalingClient: SignalingClient
    private lateinit var webRTCResponder: WebRTCResponder
    private lateinit var dataChannelBridge: DataChannelBridge

    @Volatile
    var isRunning: Boolean = false
        private set

    @Volatile
    var isProcessing: Boolean = false
        private set

    val currentRequests: Int get() = if (isProcessing) 1 else 0

    fun start() {
        if (isRunning) return
        isRunning = true

        scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

        dataChannelBridge = DataChannelBridge(engine, modelId)
        signalingClient = SignalingClient(signalingUrl, apiKey, scope!!)
        webRTCResponder = WebRTCResponder(signalingClient, dataChannelBridge, scope!!)

        signalingClient.connect()
        scope!!.launch { listenForSignalingMessages() }
    }

    private suspend fun listenForSignalingMessages() {
        for (message in signalingClient.incomingMessages) {
            when (message) {
                is SignalingClient.SignalingMessage.Connected -> {
                    signalingClient.register(
                        modelId = modelId,
                        vramMb = vramMb,
                        backend = backend
                    )
                }
                is SignalingClient.SignalingMessage.Offer -> {
                    isProcessing = true
                    webRTCResponder.handleOffer(message.sdp, message.roomId) { _ ->
                        isProcessing = false
                    }
                }
                is SignalingClient.SignalingMessage.IceCandidate -> {
                    webRTCResponder.handleIceCandidate(
                        candidate = message.candidate,
                        sdpMid = message.sdpMid,
                        sdpMLineIndex = message.sdpMLineIndex,
                        roomId = message.roomId
                    )
                }
                is SignalingClient.SignalingMessage.Error -> {
                    // Log error
                }
                is SignalingClient.SignalingMessage.Disconnected -> {
                    // Will auto-reconnect from SignalingClient
                }
            }
        }
    }

    fun stop() {
        isRunning = false
        signalingClient.disconnect()
        webRTCResponder.cleanupAll()
        scope?.cancel()
        scope = null
    }
}
