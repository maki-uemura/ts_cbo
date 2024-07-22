https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

所感      Name                         Quant method    Bits    Size       Max RAM required    Use case
実用不可  llama-2-7b-chat.Q2_K.gguf    Q2_K               2    2.83 GB    5.33 GB             smallest, significant quality loss - not recommended for most purposes
          llama-2-7b-chat.Q3_K_S.gguf  Q3_K_S             3    2.95 GB    5.45 GB             very small, high quality loss
          llama-2-7b-chat.Q3_K_M.gguf  Q3_K_M             3    3.30 GB    5.80 GB             very small, high quality loss
          llama-2-7b-chat.Q3_K_L.gguf  Q3_K_L             3    3.60 GB    6.10 GB             small, substantial quality loss
          llama-2-7b-chat.Q4_0.gguf    Q4_0               4    3.83 GB    6.33 GB             legacy; small, very high quality loss - prefer using Q3_K_M
          llama-2-7b-chat.Q4_K_S.gguf  Q4_K_S             4    3.86 GB    6.36 GB             small, greater quality loss
          llama-2-7b-chat.Q4_K_M.gguf  Q4_K_M             4    4.08 GB    6.58 GB             medium, balanced quality - recommended
          llama-2-7b-chat.Q5_0.gguf    Q5_0               5    4.65 GB    7.15 GB             legacy; medium, balanced quality - prefer using Q4_K_M
          llama-2-7b-chat.Q5_K_S.gguf  Q5_K_S             5    4.65 GB    7.15 GB             large, low quality loss - recommended
試す      llama-2-7b-chat.Q5_K_M.gguf  Q5_K_M             5    4.78 GB    7.28 GB             large, very low quality loss - recommended
          llama-2-7b-chat.Q6_K.gguf    Q6_K               6    5.53 GB    8.03 GB             very large, extremely low quality loss
          llama-2-7b-chat.Q8_0.gguf    Q8_0               8    7.16 GB    9.66 GB             very large, extremely low quality loss - not recommended



https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b
14GB
