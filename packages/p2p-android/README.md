# p2p-connector-android

Biblioteca Kotlin open-source (MIT) para convertir cualquier app Android con LiteRT-LM en un **provider de inferencia P2P** dentro de la red Friendly LLM.

## Cómo funciona

```
App Android                  Signaling Server                Cliente Web
 ┌─────────────┐            ┌────────────────┐            ┌────────────┐
 │  LiteRT-LM   │            │  WebSocket      │            │  Browser    │
 │  Engine      │◄───────────│  room mgmt      │◄───────────│  WebRTC     │
 │  (gemma-4)   │  WebRTC   │  ICE relay      │  WebRTC    │  Chat UI    │
 └─────────────┘ DataChannel└────────────────┘ DataChannel └────────────┘
```

## Instalación

```kotlin
// build.gradle.kts (app)
dependencies {
    implementation("dev.friendlyllm:p2p-connector-android:0.1.0")
}
```

O desde source:

```kotlin
// settings.gradle.kts
include(":p2p-connector-android")
project(":p2p-connector-android").projectDir = file("../litert-ollama/packages/p2p-android")
```

## Uso básico

```kotlin
// 1. Implementa InferenceEngine (conecta a tu LiteRT-LM Engine)
val engine = object : InferenceEngine {
    override suspend fun sendMessageAsync(
        request: IncomingRequest,
        onToken: (String) -> Unit
    ): String? {
        // Usa LiteRT-LM Conversation.sendMessageAsync()
        val conversation = liteRtEngine.createConversation(...)
        conversation.sendMessageAsync(Contents.of(request.messages.joinToString("\n"))) { chunk ->
            onToken(chunk.text)
        }
        return null
    }
}

// 2. Crea el connector
val connector = P2PConnectorEngine(
    signalingUrl = "wss://signal.friendlyllm.com/ws",
    apiKey = "sk-...",  // API key del provider
    engine = engine,
    modelId = "gemma-4-E2B-it",
    vramMb = 4096,
    backend = "gpu"
)

// 3. Inicia
connector.start()

// 4. Cuando quieras detener
connector.stop()
```

## UI Toggle (opcional)

```kotlin
// En SettingsScreen.kt
@Composable
fun ProviderSettings(
    connector: P2PConnectorEngine
) {
    var enabled by remember { mutableStateOf(false) }
    
    Switch(
        checked = enabled,
        onCheckedChange = {
            enabled = it
            if (it) connector.start() else connector.stop()
        }
    )
    
    if (enabled) {
        Text("Solicitudes activas: ${connector.currentRequests}")
    }
}
```

## Arquitectura

```
P2PConnectorEngine (facade)
├── SignalingClient          — WebSocket al signaling server
├── WebRTCResponder          — PeerConnection + ICE + DataChannel
├── DataChannelBridge        — Traduce DataChannel ↔ InferenceEngine
└── InferenceEngine (interface) — Lo implementa el host (LiteRT-LM, etc.)
```

## Licencia

MIT — libre para usar, modificar y distribuir.
