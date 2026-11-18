# Monitoramento de crédito · Data Drift

O controle gerou **0 alertas em 20 variáveis**. No lote alterado, **2 de 20** ultrapassaram os limites de drift. O limite de crédito aumentou em 50% e o valor do primeiro pagamento caiu pela metade.

As duas comparações usam os mesmos 18.000 registros de referência. Cada lote tem 6.000 registros. A tabela traz os resultados das variáveis sinalizadas no lote alterado.

| Variável | PSI observado | Limite PSI | KS D observado | Limite KS D |
|---|---:|---:|---:|---:|
| `LIMIT_BAL` | 0,382789 | 0,006938 | 0,212167 | 0,028279 |
| `PAY_AMT1` | 0,458898 | 0,008578 | 0,256611 | 0,026194 |

O alerta exige que PSI e KS D ultrapassem os limites calibrados para a variável e que o valor de p corrigido por Holm seja de até 0,05. O adaptador Evidently registrou as mesmas decisões para as 20 variáveis de cada comparação.

**[Baixar o relatório HTML completo](relatorio.html?raw=1)**. Abra o arquivo no navegador para consultar as 40 comparações, os limites e os valores de p ajustados. O relatório funciona offline.

Uma mudança na distribuição indica onde investigar. Para medir o impacto no desempenho, o projeto também analisa um cenário com mudança nos rótulos. [Análise de Concept Drift](../../README.md#concept-drift)
