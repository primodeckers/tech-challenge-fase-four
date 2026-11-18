# Vídeo de monitoramento de crédito

A composição `EntregaSTAR` tem oito cenas a 24 quadros por segundo. `tempos.json` guarda o primeiro quadro de cada cena e o quadro final. As trocas acompanham o início das frases na narração em português.

A montagem usa dois arquivos da produção local em `artifacts/video-s12/public/`:

- `narracao.mp3` — locução desta edição, gerada no ElevenLabs.
- `mlflow.png` — captura da execução de Data Drift, com 6.000 registros e dois alertas.

Na pasta `video`, execute `npm install` e `npm run studio` para visualizar a composição com os arquivos acima. `npm run render` escreve o MP4 intermediário em `artifacts/video-s12/`. O áudio começa no quadro 12. Ao substituir a locução, atualize os cortes e a duração total usando as marcações da nova gravação.

A edição final inclui legendas selecionáveis em português e capítulos. O arquivo SRT também pode ser importado separadamente no YouTube Studio.
