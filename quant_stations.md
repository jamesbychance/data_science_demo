# Quant Stations

The book describes the structure of a modern asset management firm using the metaphor of an **assembly line** or **production chain**, where highly specialised teams operate distinct "stations". This approach, known as the **meta-strategy paradigm**, is designed to yield scientific discoveries at a predictable rate through teamwork.

The five primary stations involved in this production chain are:

1.  **Data Curators**
    *   **Role:** This station is responsible for **collecting, cleaning, indexing, storing, adjusting, and delivering all data** to the production chain.
    *   **Expertise:** Team members are experts in **market microstructure** and data protocols, such as FIX.
    *   **Relevance:** The book discusses only a few aspects of data curation in Chapter 1.

2.  **Feature Analysts**
    *   **Role:** This critical station transforms raw data into **informative signals** that possess predictive power over financial variables. They produce and catalogue libraries of findings that can be useful to a multiplicity of other stations.
    *   **Expertise:** They are experts in information theory, signal extraction and processing, **labelling**, weighting, classifiers, and feature importance techniques.
    *   **Relevance:** The current project steps—Data Structuring (Phase 0) and Labelling (Phase 1, Chapter 3)—fall under the purview of this station. Chapters 2 through 9 and Chapters 17 through 19 are dedicated to the work of Feature Analysts.

3.  **Strategists**
    *   **Role:** This station **transforms informative features into actual investment algorithms**. The strategists formulate a general theory explaining the economic mechanism that causes an agent to lose money to the firm, and the investment strategy serves as the experiment to test this theory.
    *   **Expertise:** Team members are data scientists with deep knowledge of financial markets and the economy.
    *   **Relevance:** Chapters 10 and 16 are dedicated to this station.

4.  **Backtesters**
    *   **Role:** This station **assesses the profitability** of an investment strategy under various scenarios, including simulating historical performance. Crucially, the backtester's analysis must evaluate the **probability of backtest overfitting**.
    *   **Process:** Backtest results are communicated only to management and are not shared with other stations.
    *   **Relevance:** Chapters 11 through 16 discuss the analyses carried out by this station.

5.  **Deployment Team**
    *   **Role:** This team **integrates the strategy code into the production line** and ensures the deployed solution is logically identical to the prototype received.
    *   **Expertise:** They are algorithm specialists and mathematical programmers who optimize implementation for minimal production latency, relying heavily on **vectorization, multiprocessing, and parallel computing techniques**.
    *   **Relevance:** Chapters 20 through 22 cover various aspects relevant to this station.
