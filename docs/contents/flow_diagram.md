# Flow diagram

The scrollable diagram below outlines the sequence of functions in `chatter`, from data cleaning to visualization.

```{eval-rst}
.. mermaid::

    ---
    config:
        layout: elk
        look: neo
        theme: neutral
        themeVariables:
            fontSize: '24px'
    ---

    flowchart TB
        classDef viridis1 fill:#440154,stroke:#333,stroke-width:1px,color:#fff
        classDef viridis2 fill:#3b528b,stroke:#333,stroke-width:1px,color:#fff
        classDef viridis3 fill:#21918c,stroke:#333,stroke-width:1px,color:#fff
        classDef viridis4 fill:#5ec962,stroke:#333,stroke-width:1px,color:#000
        classDef viridis5 fill:#fde725,stroke:#333,stroke-width:1px,color:#000

        subgraph subgraph1["Audio preprocessing"]
            direction TB
            C("Gain normalization with *librosa*")
            B("High/low-pass filtering with *pydub*")
            DD("Noise reduction with *biodenoising*")
            D("Noise reduction with *noisereduce*")
            E("Compression with *audiocomplib*")
            F("Peak limiting with *audiocomplib*")
        end
        
        subgraph subgraph2["Segmentation"]
            I("Amplitude-based segmentation with librosa")
            J("Image-based segmentation with *Pykanto*")
        end
        
        subgraph subgraph3["Spectrogram preprocessing"]
            M("Normalization")
            N("Centering/padding")
            O("Downsampling to 128x128")
        end
        
        subgraph subgraph4["Variational autoencoder in *pytorch*"]
            R("Latent space")
            Q("Encoder layers")
            S("Decoder layers")
        end
        
        subgraph subgraph5["Analysis"]
            V("Complexity using cosine distances within sequences")
            X("Dimensionality reduction with *pacmap*")
            Y("Birch clustering with *scikit*")
            Z("Predictability using vector autoregression in *numpy*")
            AA("Novelty using density estimation in *denmarf*")
        end

        A["Raw audio files"] --> B & H("Species identification with *birdnetlib*")
        H --> B
        B --> C
        C --> DD
        DD --> D
        D --> E
        E --> F
        F --> G["Processed audio files"]
        G --> I & J
        L["Annotations from external segmentation"] --> K["Raw spectrograms"]
        I --> K
        J --> K
        K --> M
        M --> N
        N --> O
        O --> P["Processed spectrograms"]
        P --> CC("Computer vision model in *transformers*")
        CC --> U
        P --> Q
        Q --> R
        R --> S & U["Embeddings"]
        S --> T["Simulated spectrograms"]
        U --> V & X & Y & Z & AA
        V --> BB["Output and visualization"]
        X --> BB
        Y --> BB
        Z --> BB
        AA --> BB

        class A,H,B,C,D,DD,E,F viridis1
        class G,I,J,L,K viridis2
        class M,N,O,P viridis3
        class Q,R,S,T,U,CC viridis4
        class V,X,Y,Z,AA,BB viridis5
```
