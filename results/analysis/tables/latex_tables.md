# Benchmark LaTeX Tables

Copy the table code below into your Overleaf document.

## Baseline Table

```latex
% ~~~~~~~~~~~~~~~~~~~~~~~~ TABLE: BASELINE ~~~~~~~~~~~~~~~~~~~~~~~~
\begin{table*}[!t]
    \caption{Performance under \textbf{Zero-Shot (Baseline)}.
    Best in \textbf{bold}, second-best \underline{underlined}.}
    \label{tab:baseline}
    \centering
    \small
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{llcccccccccccccc}
        \toprule
         & Model & \multicolumn{2}{c}{AARIZ} & \multicolumn{1}{c}{BRAR} & \multicolumn{2}{c}{CODE} & \multicolumn{3}{c}{DENPAR} & \multicolumn{2}{c}{DENTALCARIES} & \multicolumn{1}{c}{DR} & \multicolumn{3}{c}{METADENT} \\
        \cmidrule(lr){3-4} \cmidrule(lr){5-5} \cmidrule(lr){6-7} \cmidrule(lr){8-10} \cmidrule(lr){11-12} \cmidrule(lr){13-13} \cmidrule(lr){14-16}
         &  & CVM (Acc) ($\uparrow$) & VQA (Acc) ($\uparrow$) & BRAR (Acc) ($\uparrow$) & Cls (Acc) ($\uparrow$) & Report (METEOR) ($\uparrow$) & Arch (Acc) ($\uparrow$) & Count (Acc) ($\uparrow$) & Site (Acc) ($\uparrow$) & Cls (Acc) ($\uparrow$) & Detect (Acc) ($\uparrow$) & DR Cls (F1w) ($\uparrow$) & Cap (BERT-F1) ($\uparrow$) & Cls (F1) ($\uparrow$) & VQA (Acc) ($\uparrow$) \\
        \midrule \multirow{3}{*}{\shortstack{Large\\VLMs}}
            & Lingshu-32B & \underline{0.127} & 0.256 & 0.490 & 0.507 & \textbf{0.269} & 0.595 & 0.350 & \textbf{0.565} & 0.049 & 0.560 & 0.596 & 0.177 & \underline{0.340} & 0.632 \\
            & MedMO-8B-Next & 0.048 & 0.214 & 0.255 & 0.300 & 0.060 & 0.610 & 0.230 & 0.365 & \underline{0.748} & 0.586 & 0.529 & 0.089 & 0.079 & 0.487 \\
            & Qwen2.5-VL-7B-Instruct & 0.000 & 0.200 & 0.268 & 0.520 & 0.173 & 0.400 & 0.360 & 0.440 & 0.067 & \underline{0.634} & 0.319 & 0.149 & 0.232 & 0.445 \\
        \cmidrule{2-16}
             & \textit{Mean} & \textit{0.058} & \textit{0.223} & \textit{0.338} & \textit{0.442} & \textit{0.167} & \textit{0.535} & \textit{0.313} & \textit{0.457} & \textit{0.288} & \textit{0.593} & \textit{0.481} & \textit{0.138} & \textit{0.217} & \textit{0.521} \\
        \cmidrule{2-16}
             & \textit{Best} & \textit{0.127} & \textit{0.256} & \textit{0.490} & \textit{0.520} & \textit{0.269} & \textit{0.610} & \textit{0.360} & \textit{0.565} & \textit{0.748} & \textit{0.634} & \textit{0.596} & \textit{0.177} & \textit{0.340} & \textit{0.632} \\
        \midrule \multirow{12}{*}{\shortstack{Compact\\VLMs}}
            & gemini-2.5-flash & 0.119 & 0.225 & 0.272 & \textbf{0.597} & 0.018 & \textbf{0.990} & 0.555 & \underline{0.560} & 0.066 & 0.543 & \textbf{0.622} & 0.144 & 0.237 & \underline{0.656} \\
            & Qwen3.5-4B & 0.040 & 0.175 & 0.174 & 0.192 & 0.111 & 0.395 & 0.025 & 0.300 & 0.053 & 0.495 & 0.542 & 0.105 & 0.157 & \textbf{0.817} \\
            & Qwen3-VL-4B-Instruct & 0.079 & 0.225 & 0.443 & 0.555 & \underline{0.219} & 0.440 & \textbf{0.640} & 0.400 & 0.527 & 0.627 & 0.237 & \textbf{0.205} & 0.222 & 0.578 \\
            & gemma-4-E4B-it & 0.040 & 0.308 & 0.557 & 0.531 & 0.150 & 0.395 & 0.555 & 0.520 & 0.217 & 0.432 & 0.615 & 0.180 & 0.308 & 0.591 \\
            & dentalgemma-1.5-4b-it & 0.032 & 0.381 & 0.376 & 0.366 & 0.155 & 0.605 & 0.020 & 0.280 & 0.058 & 0.503 & 0.073 & 0.000 & 0.194 & 0.165 \\
            & medgemma-4b-it & 0.032 & \textbf{0.397} & 0.443 & 0.330 & 0.154 & 0.395 & 0.355 & 0.395 & 0.058 & 0.518 & 0.570 & 0.144 & 0.164 & 0.542 \\
            & paligemma2-3b-mix-448 & 0.032 & 0.203 & 0.103 & 0.000 & 0.025 & 0.000 & 0.350 & 0.350 & 0.000 & \textbf{0.640} & 0.000 & — & 0.000 & — \\
            & SmolVLM2-2.2B-Instruct & 0.048 & 0.225 & \underline{0.564} & 0.205 & 0.170 & 0.605 & 0.355 & 0.250 & \textbf{0.863} & 0.439 & 0.556 & 0.097 & 0.153 & — \\
            & InternVL3\_5-2B-HF & 0.000 & 0.373 & 0.503 & 0.170 & 0.021 & 0.395 & 0.000 & 0.270 & 0.058 & 0.361 & 0.092 & 0.000 & 0.000 & 0.147 \\
            & gemma-4-E2B-it & 0.032 & \underline{0.389} & \underline{0.564} & 0.514 & 0.147 & 0.270 & 0.440 & 0.400 & 0.058 & 0.615 & 0.245 & 0.146 & 0.246 & 0.477 \\
            & InternVL3\_5-1B-HF & 0.071 & 0.210 & 0.262 & 0.206 & 0.026 & 0.280 & 0.090 & 0.355 & 0.062 & 0.611 & \underline{0.621} & 0.000 & 0.000 & 0.338 \\
            & gemini-2.0-flash & \textbf{0.254} & 0.294 & \textbf{0.570} & \underline{0.560} & 0.158 & \underline{0.840} & \underline{0.615} & 0.545 & 0.062 & 0.500 & — & \underline{0.182} & \textbf{0.358} & 0.627 \\
        \cmidrule{2-16}
             & \textit{Mean} & \textit{0.065} & \textit{0.284} & \textit{0.403} & \textit{0.352} & \textit{0.113} & \textit{0.468} & \textit{0.333} & \textit{0.385} & \textit{0.173} & \textit{0.524} & \textit{0.379} & \textit{0.109} & \textit{0.170} & \textit{0.494} \\
        \cmidrule{2-16}
             & \textit{Best} & \textit{0.254} & \textit{0.397} & \textit{0.570} & \textit{0.597} & \textit{0.219} & \textit{0.990} & \textit{0.640} & \textit{0.560} & \textit{0.863} & \textit{0.640} & \textit{0.622} & \textit{0.205} & \textit{0.358} & \textit{0.817} \\
        \bottomrule
    \end{tabular}
\end{table*}
```

## 1shot Table

```latex
% ~~~~~~~~~~~~~~~~~~~~~~~~ TABLE: 1SHOT ~~~~~~~~~~~~~~~~~~~~~~~~
\begin{table*}[!t]
    \caption{Performance under \textbf{One-Shot}.
    Best in \textbf{bold}, second-best \underline{underlined}.}
    \label{tab:1shot}
    \centering
    \small
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{llcccccccccccccc}
        \toprule
         & Model & \multicolumn{2}{c}{AARIZ} & \multicolumn{1}{c}{BRAR} & \multicolumn{2}{c}{CODE} & \multicolumn{3}{c}{DENPAR} & \multicolumn{2}{c}{DENTALCARIES} & \multicolumn{1}{c}{DR} & \multicolumn{3}{c}{METADENT} \\
        \cmidrule(lr){3-4} \cmidrule(lr){5-5} \cmidrule(lr){6-7} \cmidrule(lr){8-10} \cmidrule(lr){11-12} \cmidrule(lr){13-13} \cmidrule(lr){14-16}
         &  & CVM (Acc) ($\uparrow$) & VQA (Acc) ($\uparrow$) & BRAR (Acc) ($\uparrow$) & Cls (Acc) ($\uparrow$) & Report (METEOR) ($\uparrow$) & Arch (Acc) ($\uparrow$) & Count (Acc) ($\uparrow$) & Site (Acc) ($\uparrow$) & Cls (Acc) ($\uparrow$) & Detect (Acc) ($\uparrow$) & DR Cls (F1w) ($\uparrow$) & Cap (BERT-F1) ($\uparrow$) & Cls (F1) ($\uparrow$) & VQA (Acc) ($\uparrow$) \\
        \midrule \multirow{3}{*}{\shortstack{Large\\VLMs}}
            & Lingshu-32B & \textbf{0.468} & 0.403 & 0.195 & \underline{0.540} & \textbf{0.292} & 0.660 & 0.340 & 0.355 & 0.556 & 0.403 & 0.570 & 0.180 & 0.215 & 0.616 \\
            & MedMO-8B-Next & 0.325 & 0.194 & 0.174 & 0.307 & \underline{0.280} & 0.605 & 0.200 & 0.350 & \textbf{0.863} & 0.583 & 0.556 & 0.157 & 0.204 & 0.550 \\
            & Qwen2.5-VL-7B-Instruct & 0.238 & 0.281 & 0.302 & 0.513 & 0.168 & 0.395 & 0.265 & 0.405 & 0.060 & \underline{0.601} & 0.242 & 0.174 & 0.237 & 0.550 \\
        \cmidrule{2-16}
             & \textit{Mean} & \textit{0.344} & \textit{0.293} & \textit{0.224} & \textit{0.453} & \textit{0.247} & \textit{0.553} & \textit{0.268} & \textit{0.370} & \textit{0.493} & \textit{0.529} & \textit{0.456} & \textit{0.170} & \textit{0.219} & \textit{0.572} \\
        \cmidrule{2-16}
             & \textit{Best} & \textit{0.468} & \textit{0.403} & \textit{0.302} & \textit{0.540} & \textit{0.292} & \textit{0.660} & \textit{0.340} & \textit{0.405} & \textit{0.863} & \textit{0.601} & \textit{0.570} & \textit{0.180} & \textit{0.237} & \textit{0.616} \\
        \midrule \multirow{11}{*}{\shortstack{Compact\\VLMs}}
            & gemini-2.5-flash & 0.194 & 0.280 & 0.304 & 0.534 & 0.030 & \textbf{0.965} & \textbf{0.610} & \textbf{0.603} & 0.124 & 0.548 & \textbf{0.665} & — & — & — \\
            & Qwen3.5-4B & 0.175 & 0.173 & 0.174 & 0.187 & 0.142 & 0.395 & 0.025 & 0.350 & 0.080 & 0.409 & 0.541 & 0.123 & 0.227 & \textbf{0.737} \\
            & Qwen3-VL-4B-Instruct & 0.429 & 0.397 & 0.195 & 0.518 & 0.245 & 0.605 & 0.465 & 0.440 & \textbf{0.863} & 0.510 & 0.347 & \textbf{0.216} & \underline{0.319} & 0.609 \\
            & gemma-4-E4B-it & 0.460 & \textbf{0.417} & 0.510 & 0.508 & 0.153 & 0.365 & \underline{0.580} & \underline{0.555} & 0.854 & 0.529 & \underline{0.656} & 0.190 & \underline{0.319} & 0.585 \\
            & dentalgemma-1.5-4b-it & \textbf{0.468} & 0.376 & 0.383 & 0.273 & 0.126 & 0.605 & 0.350 & 0.350 & 0.854 & 0.487 & 0.088 & 0.015 & 0.129 & 0.543 \\
            & medgemma-4b-it & 0.357 & \underline{0.405} & 0.174 & 0.305 & 0.168 & 0.605 & 0.355 & 0.370 & 0.836 & 0.471 & 0.609 & 0.171 & 0.193 & 0.566 \\
            & SmolVLM2-2.2B-Instruct & 0.048 & 0.225 & 0.248 & 0.236 & 0.255 & 0.605 & 0.355 & 0.260 & \textbf{0.863} & 0.361 & 0.556 & 0.128 & 0.215 & 0.575 \\
            & InternVL3\_5-2B-HF & 0.048 & 0.202 & \textbf{0.570} & 0.207 & 0.029 & 0.395 & 0.005 & 0.360 & 0.062 & 0.475 & 0.122 & 0.000 & 0.126 & 0.236 \\
            & gemma-4-E2B-it & 0.048 & 0.383 & \textbf{0.570} & 0.423 & 0.148 & 0.370 & 0.500 & 0.410 & 0.637 & \textbf{0.602} & 0.228 & 0.179 & 0.255 & 0.563 \\
            & InternVL3\_5-1B-HF & 0.095 & 0.210 & 0.174 & 0.263 & 0.015 & 0.605 & 0.285 & 0.350 & 0.080 & 0.368 & 0.610 & 0.000 & 0.040 & 0.272 \\
            & gemini-2.0-flash & 0.444 & 0.360 & 0.456 & \textbf{0.562} & 0.168 & \underline{0.960} & 0.570 & 0.515 & 0.076 & 0.592 & — & \underline{0.197} & \textbf{0.362} & \underline{0.639} \\
        \cmidrule{2-16}
             & \textit{Mean} & \textit{0.251} & \textit{0.312} & \textit{0.342} & \textit{0.365} & \textit{0.134} & \textit{0.589} & \textit{0.373} & \textit{0.415} & \textit{0.484} & \textit{0.487} & \textit{0.442} & \textit{0.122} & \textit{0.218} & \textit{0.532} \\
        \cmidrule{2-16}
             & \textit{Best} & \textit{0.468} & \textit{0.417} & \textit{0.570} & \textit{0.562} & \textit{0.255} & \textit{0.965} & \textit{0.610} & \textit{0.603} & \textit{0.863} & \textit{0.602} & \textit{0.665} & \textit{0.216} & \textit{0.362} & \textit{0.737} \\
        \bottomrule
    \end{tabular}
\end{table*}
```

## 2shot Table

```latex
% ~~~~~~~~~~~~~~~~~~~~~~~~ TABLE: 2SHOT ~~~~~~~~~~~~~~~~~~~~~~~~
\begin{table*}[!t]
    \caption{Performance under \textbf{Two-Shot}.
    Best in \textbf{bold}, second-best \underline{underlined}.}
    \label{tab:2shot}
    \centering
    \small
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{llcccccccccccccc}
        \toprule
         & Model & \multicolumn{2}{c}{AARIZ} & \multicolumn{1}{c}{BRAR} & \multicolumn{2}{c}{CODE} & \multicolumn{3}{c}{DENPAR} & \multicolumn{2}{c}{DENTALCARIES} & \multicolumn{1}{c}{DR} & \multicolumn{3}{c}{METADENT} \\
        \cmidrule(lr){3-4} \cmidrule(lr){5-5} \cmidrule(lr){6-7} \cmidrule(lr){8-10} \cmidrule(lr){11-12} \cmidrule(lr){13-13} \cmidrule(lr){14-16}
         &  & CVM (Acc) ($\uparrow$) & VQA (Acc) ($\uparrow$) & BRAR (Acc) ($\uparrow$) & Cls (Acc) ($\uparrow$) & Report (METEOR) ($\uparrow$) & Arch (Acc) ($\uparrow$) & Count (Acc) ($\uparrow$) & Site (Acc) ($\uparrow$) & Cls (Acc) ($\uparrow$) & Detect (Acc) ($\uparrow$) & DR Cls (F1w) ($\uparrow$) & Cap (BERT-F1) ($\uparrow$) & Cls (F1) ($\uparrow$) & VQA (Acc) ($\uparrow$) \\
        \midrule \multirow{3}{*}{\shortstack{Large\\VLMs}}
            & Lingshu-32B & \textbf{0.468} & 0.356 & 0.383 & \underline{0.543} & \textbf{0.301} & 0.840 & 0.390 & 0.535 & 0.087 & 0.457 & 0.556 & 0.200 & 0.093 & \underline{0.644} \\
            & MedMO-8B-Next & 0.230 & 0.146 & \textbf{0.564} & 0.390 & \underline{0.277} & 0.605 & 0.340 & 0.370 & \underline{0.858} & 0.645 & 0.556 & 0.151 & 0.198 & 0.557 \\
            & Qwen2.5-VL-7B-Instruct & 0.103 & 0.248 & 0.342 & 0.510 & 0.176 & 0.395 & 0.240 & 0.340 & 0.058 & \textbf{0.670} & 0.265 & 0.153 & 0.214 & 0.555 \\
        \cmidrule{2-16}
             & \textit{Mean} & \textit{0.267} & \textit{0.250} & \textit{0.430} & \textit{0.481} & \textit{0.251} & \textit{0.613} & \textit{0.323} & \textit{0.415} & \textit{0.334} & \textit{0.591} & \textit{0.459} & \textit{0.168} & \textit{0.168} & \textit{0.585} \\
        \cmidrule{2-16}
             & \textit{Best} & \textit{0.468} & \textit{0.356} & \textit{0.564} & \textit{0.543} & \textit{0.301} & \textit{0.840} & \textit{0.390} & \textit{0.535} & \textit{0.858} & \textit{0.670} & \textit{0.556} & \textit{0.200} & \textit{0.214} & \textit{0.644} \\
        \midrule \multirow{11}{*}{\shortstack{Compact\\VLMs}}
            & gemini-2.5-flash & 0.227 & 0.237 & 0.291 & 0.529 & 0.040 & \textbf{0.976} & \underline{0.597} & 0.584 & — & — & \textbf{0.681} & — & — & — \\
            & Qwen3.5-4B & 0.159 & 0.175 & 0.208 & 0.191 & 0.121 & 0.395 & 0.025 & 0.450 & 0.084 & 0.452 & 0.533 & 0.118 & \textbf{0.398} & \textbf{0.773} \\
            & Qwen3-VL-4B-Instruct & 0.262 & 0.249 & 0.309 & \textbf{0.593} & 0.271 & 0.615 & 0.510 & 0.460 & 0.469 & 0.588 & 0.245 & \textbf{0.210} & 0.283 & 0.586 \\
            & gemma-4-E4B-it & 0.048 & \textbf{0.411} & 0.403 & 0.540 & 0.160 & 0.415 & 0.585 & \textbf{0.585} & 0.761 & 0.497 & \underline{0.636} & 0.181 & 0.332 & 0.599 \\
            & dentalgemma-1.5-4b-it & 0.111 & 0.343 & 0.369 & 0.360 & 0.095 & 0.605 & 0.150 & 0.250 & 0.062 & 0.433 & 0.088 & 0.018 & 0.145 & 0.542 \\
            & medgemma-4b-it & 0.254 & 0.208 & \textbf{0.564} & 0.413 & 0.236 & 0.385 & 0.135 & 0.395 & 0.058 & 0.497 & 0.618 & 0.166 & 0.136 & 0.554 \\
            & SmolVLM2-2.2B-Instruct & 0.048 & 0.225 & \textbf{0.564} & — & — & 0.605 & 0.355 & 0.375 & \textbf{0.863} & 0.565 & 0.556 & 0.106 & 0.160 & 0.536 \\
            & InternVL3\_5-2B-HF & 0.048 & 0.216 & \textbf{0.564} & 0.208 & 0.031 & 0.395 & 0.000 & 0.310 & 0.058 & 0.455 & 0.138 & 0.000 & 0.131 & 0.345 \\
            & gemma-4-E2B-it & 0.048 & 0.217 & \textbf{0.564} & 0.507 & 0.152 & 0.325 & 0.330 & 0.365 & 0.058 & 0.592 & 0.228 & 0.174 & 0.188 & 0.558 \\
            & InternVL3\_5-1B-HF & 0.238 & 0.210 & \textbf{0.564} & 0.231 & 0.025 & 0.415 & 0.075 & 0.350 & 0.080 & \underline{0.651} & 0.574 & 0.000 & 0.085 & 0.322 \\
            & gemini-2.0-flash & \underline{0.325} & \underline{0.387} & 0.369 & 0.538 & 0.177 & \underline{0.945} & \textbf{0.625} & \textbf{0.585} & 0.071 & 0.605 & — & \underline{0.202} & \underline{0.353} & 0.634 \\
        \cmidrule{2-16}
             & \textit{Mean} & \textit{0.161} & \textit{0.262} & \textit{0.434} & \textit{0.411} & \textit{0.131} & \textit{0.552} & \textit{0.308} & \textit{0.428} & \textit{0.256} & \textit{0.533} & \textit{0.430} & \textit{0.117} & \textit{0.221} & \textit{0.545} \\
        \cmidrule{2-16}
             & \textit{Best} & \textit{0.325} & \textit{0.411} & \textit{0.564} & \textit{0.593} & \textit{0.271} & \textit{0.976} & \textit{0.625} & \textit{0.585} & \textit{0.863} & \textit{0.651} & \textit{0.681} & \textit{0.210} & \textit{0.398} & \textit{0.773} \\
        \bottomrule
    \end{tabular}
\end{table*}
```

## Sft Table

```latex
% ~~~~~~~~~~~~~~~~~~~~~~~~ TABLE: SFT ~~~~~~~~~~~~~~~~~~~~~~~~
\begin{table*}[!t]
    \caption{Performance under \textbf{Instruction Tuning (LoRA)}.
    Best in \textbf{bold}, second-best \underline{underlined}.}
    \label{tab:sft}
    \centering
    \small
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{llcccccccccccccc}
        \toprule
         & Model & \multicolumn{2}{c}{AARIZ} & \multicolumn{1}{c}{BRAR} & \multicolumn{2}{c}{CODE} & \multicolumn{3}{c}{DENPAR} & \multicolumn{2}{c}{DENTALCARIES} & \multicolumn{1}{c}{DR} & \multicolumn{3}{c}{METADENT} \\
        \cmidrule(lr){3-4} \cmidrule(lr){5-5} \cmidrule(lr){6-7} \cmidrule(lr){8-10} \cmidrule(lr){11-12} \cmidrule(lr){13-13} \cmidrule(lr){14-16}
         &  & CVM (Acc) ($\uparrow$) & VQA (Acc) ($\uparrow$) & BRAR (Acc) ($\uparrow$) & Cls (Acc) ($\uparrow$) & Report (METEOR) ($\uparrow$) & Arch (Acc) ($\uparrow$) & Count (Acc) ($\uparrow$) & Site (Acc) ($\uparrow$) & Cls (Acc) ($\uparrow$) & Detect (Acc) ($\uparrow$) & DR Cls (F1w) ($\uparrow$) & Cap (BERT-F1) ($\uparrow$) & Cls (F1) ($\uparrow$) & VQA (Acc) ($\uparrow$) \\
        \midrule \multirow{3}{*}{\shortstack{Large\\VLMs}}
            & Lingshu-32B & \underline{0.468} & \underline{0.590} & \underline{0.577} & \textbf{0.818} & \textbf{0.567} & 0.805 & 0.340 & 0.550 & \underline{0.058} & 0.923 & 0.787 & 0.259 & 0.096 & \textbf{0.800} \\
            & MedMO-8B-Next & \underline{0.468} & 0.589 & 0.416 & 0.813 & 0.551 & 0.425 & 0.255 & 0.250 & \underline{0.058} & 0.640 & 0.608 & 0.241 & 0.000 & 0.756 \\
            & Qwen2.5-VL-7B-Instruct & \underline{0.468} & 0.589 & 0.537 & 0.803 & \underline{0.566} & 0.995 & 0.710 & 0.715 & \underline{0.058} & 0.912 & 0.773 & 0.259 & 0.093 & 0.776 \\
        \cmidrule{2-16}
             & \textit{Mean} & \textit{0.468} & \textit{0.589} & \textit{0.510} & \textit{0.811} & \textit{0.561} & \textit{0.742} & \textit{0.435} & \textit{0.505} & \textit{0.058} & \textit{0.825} & \textit{0.723} & \textit{0.253} & \textit{0.063} & \textit{0.777} \\
        \cmidrule{2-16}
             & \textit{Best} & \textit{0.468} & \textit{0.590} & \textit{0.577} & \textit{0.818} & \textit{0.567} & \textit{0.995} & \textit{0.710} & \textit{0.715} & \textit{0.058} & \textit{0.923} & \textit{0.787} & \textit{0.259} & \textit{0.096} & \textit{0.800} \\
        \midrule \multirow{10}{*}{\shortstack{Compact\\VLMs}}
            & Qwen3.5-4B & 0.160 & 0.229 & 0.201 & 0.462 & 0.444 & 0.995 & 0.470 & 0.550 & \textbf{0.509} & 0.930 & 0.557 & 0.187 & 0.246 & \underline{0.781} \\
            & Qwen3-VL-4B-Instruct & \underline{0.468} & 0.589 & 0.570 & 0.798 & 0.561 & \textbf{1.000} & \underline{0.765} & \textbf{0.870} & \underline{0.058} & \textbf{0.932} & \underline{0.834} & \textbf{0.272} & \textbf{0.286} & 0.714 \\
            & gemma-4-E4B-it & \textbf{0.500} & 0.569 & 0.564 & \underline{0.815} & 0.550 & 0.990 & 0.645 & \textbf{0.870} & \underline{0.058} & 0.903 & 0.780 & \underline{0.266} & 0.241 & 0.744 \\
            & dentalgemma-1.5-4b-it & \underline{0.468} & 0.589 & \textbf{0.584} & \underline{0.815} & 0.560 & 0.915 & 0.605 & 0.520 & \underline{0.058} & 0.916 & 0.750 & 0.259 & \underline{0.280} & 0.736 \\
            & medgemma-4b-it & \underline{0.468} & 0.589 & 0.503 & 0.803 & 0.556 & 0.990 & 0.525 & 0.615 & \underline{0.058} & 0.908 & 0.799 & 0.261 & 0.226 & 0.745 \\
            & paligemma2-3b-mix-448 & 0.467 & 0.432 & 0.297 & 0.750 & 0.462 & \textbf{1.000} & \textbf{0.795} & 0.555 & 0.049 & 0.919 & 0.581 & 0.000 & 0.003 & 0.000 \\
            & SmolVLM2-2.2B-Instruct & 0.270 & \textbf{0.594} & 0.503 & 0.785 & 0.525 & \textbf{1.000} & 0.715 & 0.740 & \underline{0.058} & 0.920 & 0.735 & 0.258 & 0.059 & 0.648 \\
            & InternVL3\_5-2B-HF & \underline{0.468} & 0.589 & 0.510 & 0.813 & 0.544 & \textbf{1.000} & 0.750 & 0.735 & \underline{0.058} & \textbf{0.932} & 0.805 & 0.262 & 0.207 & 0.647 \\
            & gemma-4-E2B-it & \underline{0.468} & 0.587 & 0.262 & 0.800 & 0.494 & 0.960 & 0.540 & 0.620 & \underline{0.058} & 0.909 & 0.743 & 0.253 & 0.246 & 0.700 \\
            & InternVL3\_5-1B-HF & \underline{0.468} & 0.589 & 0.242 & 0.800 & 0.532 & 0.990 & 0.705 & 0.685 & \underline{0.058} & 0.906 & \textbf{0.848} & — & — & 0.602 \\
        \cmidrule{2-16}
             & \textit{Mean} & \textit{0.420} & \textit{0.536} & \textit{0.424} & \textit{0.764} & \textit{0.523} & \textit{0.984} & \textit{0.651} & \textit{0.676} & \textit{0.102} & \textit{0.918} & \textit{0.743} & \textit{0.224} & \textit{0.199} & \textit{0.632} \\
        \cmidrule{2-16}
             & \textit{Best} & \textit{0.500} & \textit{0.594} & \textit{0.584} & \textit{0.815} & \textit{0.561} & \textit{1.000} & \textit{0.795} & \textit{0.870} & \textit{0.509} & \textit{0.932} & \textit{0.848} & \textit{0.272} & \textit{0.286} & \textit{0.781} \\
        \bottomrule
    \end{tabular}
\end{table*}
```

