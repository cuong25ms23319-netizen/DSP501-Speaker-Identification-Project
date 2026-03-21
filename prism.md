\documentclass[onecolumn]{IEEEtran}
\usepackage{graphicx}
\usepackage{float}
\usepackage{amsmath}
\usepackage{amssymb}
\hbadness=10000

\title{Technical Report: Speaker Identification System}

\author{
    \IEEEauthorblockN{Nguyen Huy Cuong},
    \IEEEauthorblockN{Hon Vi Dan},
    \IEEEauthorblockN{Le Nhut Thanh Quang},
    \IEEEauthorblockN{Nguyen Duc Minh Khoa}

    \IEEEauthorblockA{Supervisor: Dr. Dang Ngoc Minh Duc}
}

\begin{document}

\maketitle

\begin{abstract}
This technical report presents the development and evaluation of a Speaker Identification System, contrasting a baseline raw signal approach (Pipeline A) with a dedicated Digital Signal Processing (DSP) and feature engineering framework (Pipeline B). Speech signals are inherently non-stationary and susceptible to environmental noise, necessitating rigorous preprocessing to maintain classification integrity. Our methodology implements a custom-designed Finite Impulse Response (FIR) filter to isolate vocal bandwidths, followed by time-frequency analysis using the Short-Time Fourier Transform (STFT) to address spectral leakage and resolution trade-offs. Feature extraction is performed using Mel-Frequency Cepstral Coefficients (MFCCs) and spectral descriptors, which are subsequently classified using Support Vector Machines (SVM), Random Forest, and Convolutional Neural Networks (CNN). Experimental results, validated through 5-fold cross-validation, demonstrate that the DSP-enhanced pipeline significantly improves accuracy and Signal-to-Noise Ratio (SNR) while reducing model overfitting. The discussion concludes that while Deep Learning models possess inherent feature-learning capabilities, handcrafted DSP remains essential for computational efficiency and performance stability in noisy real-world environments.
\end{abstract}

\begin{IEEEkeywords}
speaker identification, digital signal processing, FIR filter, STFT, MFCC, machine learning, deep learning
\end{IEEEkeywords}

\section{Introduction}
\input{Section01.tex}

\section{Signal Analysis}
\input{Section02.tex}
\input{Section02_Cuong.tex}

\section{DSP Methodology}
\input{Section03.tex}

\section{Feature Engineering}
\input{Section04.tex}

\section{AI Modeling}
\input{Section05.tex}

\section{Experimental Results}
\input{Section06.tex}

\section{Comparative Analysis}
\input{Section07.tex}

\section{Discussion}
\input{Section08.tex}

\section{Limitations}
\input{Section09.tex}

\section{Conclusion}
\input{Section10.tex}

\input{Section11.tex}

\end{document}