package dev.friendlyllm.p2p

import com.google.gson.Gson
import dev.friendlyllm.p2p.models.InferenceChunk
import dev.friendlyllm.p2p.models.IncomingRequest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.webrtc.DataChannel

interface InferenceEngine {
    suspend fun sendMessageAsync(
        request: IncomingRequest,
        onToken: (String) -> Unit
    ): String?
}

class DataChannelBridge(
    private val engine: InferenceEngine,
    private val modelId: String
) {
    private val gson = Gson()

    fun handleDataChannelMessage(
        dataChannel: DataChannel,
        buffer: DataChannel.Buffer
    ) {
        val text = buffer.data.let { buf ->
            val bytes = ByteArray(buf.remaining())
            buf.get(bytes)
            String(bytes, Charsets.UTF_8)
        }

        try {
            val request = gson.fromJson(text, IncomingRequest::class.java)

            // Send first chunk immediately
            sendChunk(dataChannel, InferenceChunk(
                model = modelId,
                response = ""
            ))

            // Stream inference tokens
            engine.sendMessageAsync(
                request = request,
                onToken = { token ->
                    sendChunk(dataChannel, InferenceChunk(
                        model = modelId,
                        response = token
                    ))
                }
            )

            // Send done
            sendChunk(dataChannel, InferenceChunk(
                model = modelId,
                response = "",
                done = true,
                done_reason = "stop"
            ))
        } catch (_: Exception) {
            sendChunk(dataChannel, InferenceChunk(
                model = modelId,
                response = "",
                done = true,
                done_reason = "error"
            ))
        }
    }

    private fun sendChunk(dataChannel: DataChannel, chunk: InferenceChunk) {
        val json = gson.toJson(chunk)
        val byteBuffer = java.nio.ByteBuffer.wrap(json.toByteArray(Charsets.UTF_8))
        val buffer = DataChannel.Buffer(byteBuffer, false)
        dataChannel.send(buffer)
    }
}
