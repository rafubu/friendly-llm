package dev.friendlyllm.p2p

import kotlinx.coroutines.*
import org.webrtc.*
import java.util.UUID

class WebRTCResponder(
    private val signalingClient: SignalingClient,
    private val dataChannelBridge: DataChannelBridge,
    private val scope: CoroutineScope
) {
    private var peerConnectionFactory: PeerConnectionFactory? = null
    private var activeConnections = mutableMapOf<String, PeerConnection>()

    private val iceServers = listOf(
        PeerConnection.IceServer.builder("stun:stun.l.google.com:19302").createIceServer(),
        PeerConnection.IceServer.builder("stun:stun1.l.google.com:19302").createIceServer()
    )

    init {
        PeerConnectionFactory.initialize(
            PeerConnectionFactory.InitializationOptions.builder(/* applicationContext */ null)
                .setFieldTrials("")
                .createInitializationOptions()
        )
        peerConnectionFactory = PeerConnectionFactory.builder()
            .setVideoDecoderFactory(null)
            .setVideoEncoderFactory(null)
            .createPeerConnectionFactory()
    }

    fun createPeerConnection(): PeerConnection? {
        val pcObserver = object : PeerConnection.Observer {
            override fun onIceCandidate(candidate: IceCandidate) {
                val roomId = findRoomIdForConnection(this)
                if (roomId != null) {
                    signalingClient.sendIceCandidate(
                        candidate = candidate.sdp,
                        sdpMid = candidate.sdpMid,
                        sdpMLineIndex = candidate.sdpMLineIndex,
                        roomId = roomId
                    )
                }
            }

            override fun onIceCandidatesRemoved(candidates: Array<out IceCandidate>) {}

            override fun onSignalingChange(state: PeerConnection.SignalingState) {}

            override fun onIceConnectionChange(state: PeerConnection.IceConnectionState) {}

            override fun onIceConnectionReceivingChange(receiving: Boolean) {}

            override fun onIceGatheringChange(state: PeerConnection.IceGatheringState) {}

            override fun onAddStream(stream: MediaStream) {}

            override fun onRemoveStream(stream: MediaStream) {}

            override fun onDataChannel(dataChannel: DataChannel) {
                dataChannel.registerObserver(object : DataChannel.Observer {
                    override fun onBufferedAmountChange(previousAmount: Long) {}

                    override fun onStateChange() {}

                    override fun onMessage(buffer: DataChannel.Buffer) {
                        scope.launch {
                            dataChannelBridge.handleDataChannelMessage(dataChannel, buffer)
                        }
                    }
                })
            }

            override fun onRenegotiationNeeded() {}

            override fun onAddTrack(track: RtpReceiver, streams: Array<out MediaStream>) {}
        }

        return peerConnectionFactory?.createPeerConnection(iceServers, pcObserver)
    }

    fun handleOffer(sdp: String, roomId: String, onComplete: (PeerConnection) -> Unit) {
        val pc = createPeerConnection() ?: return
        activeConnections[roomId] = pc

        val sessionDescription = SessionDescription(SessionDescription.Type.OFFER, sdp)
        pc.setRemoteDescription(object : SdpObserver {
            override fun onCreateSuccess(sdp: SessionDescription) {}

            override fun onSetSuccess() {
                pc.createAnswer(object : SdpObserver {
                    override fun onCreateSuccess(answerSdp: SessionDescription) {
                        pc.setLocalDescription(object : SdpObserver {
                            override fun onCreateSuccess(sdp: SessionDescription) {}
                            override fun onSetSuccess() {
                                signalingClient.sendOffer(answerSdp.description, roomId)
                                onComplete(pc)
                            }
                            override fun onCreateFailure(error: String) {}
                            override fun onSetFailure(error: String) {}
                        }, answerSdp)
                    }

                    override fun onSetSuccess() {}
                    override fun onCreateFailure(error: String) {}
                    override fun onSetFailure(error: String) {}
                }, PeerConnection.RTCConfiguration.builder(iceServers).build())
            }

            override fun onCreateFailure(error: String) {}
            override fun onSetFailure(error: String) {}
        }, sessionDescription)
    }

    fun handleIceCandidate(
        candidate: String,
        sdpMid: String?,
        sdpMLineIndex: Int?,
        roomId: String
    ) {
        val pc = activeConnections[roomId]
        if (pc != null && sdpMid != null && sdpMLineIndex != null) {
            val iceCandidate = IceCandidate(sdpMid, sdpMLineIndex, candidate)
            pc.addIceCandidate(iceCandidate)
        }
    }

    fun cleanupRoom(roomId: String) {
        activeConnections[roomId]?.close()
        activeConnections.remove(roomId)
    }

    fun cleanupAll() {
        activeConnections.values.forEach { it.close() }
        activeConnections.clear()
        peerConnectionFactory?.dispose()
        peerConnectionFactory = null
    }

    private fun findRoomIdForConnection(observer: PeerConnection.Observer): String? {
        return activeConnections.entries
            .firstOrNull { (_, pc) ->
                pc.javaClass.name.contains(observer.javaClass.name)
            }?.key
    }
}
