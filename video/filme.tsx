import React from "react";
import {
  AbsoluteFill,
  Audio,
  Composition,
  Img,
  Sequence,
  interpolate,
  registerRoot,
  staticFile,
  useCurrentFrame,
} from "remotion";
import timing from "./tempos.json";
import { ConceptDrift } from "./ConceptDrift";
const mint = "#69dfc5",
  amber = "#ffa56f",
  ink = "#f2f5ed",
  muted = "#b5c9c9";
const ramp = (f: number, a: number, b: number) =>
  interpolate(f, [a, b], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
const labels = [
  "SITUAÇÃO",
  "TAREFA",
  "AÇÃO",
  "AÇÃO",
  "AÇÃO",
  "AÇÃO",
  "AÇÃO",
  "RESULTADO",
];
const titles = [
  "Quando o risco muda",
  "Três conjuntos, três papéis",
  "Qualidade antes da inferência",
  "Duas mudanças na distribuição",
  "",
  "O que aconteceu com o lote?",
  "Quem recebe mais falsos positivos?",
  "Do sinal à investigação",
];
const notes = [
  "Base histórica UCI • Default of Credit Card Clients",
  "Separação aleatória estratificada • Identificadores disjuntos",
  "Contrato Pandera • Modelo congelado após calibração",
  "20 variáveis monitoradas • PSI, KS e correção de Holm",
  "",
  "MLflow local • Cada execução possui um identificador próprio",
  "Controle histórico • Mesmo modelo e limiar em todos os grupos",
  "FIAP • Machine Learning Engineering • Fase 4",
];
function Text({ x, y, size = 26, color = ink, children, bold = false }: any) {
  return (
    <text
      x={x}
      y={y}
      fill={color}
      fontSize={size}
      fontWeight={bold ? 650 : 400}
    >
      {children}
    </text>
  );
}
function Stat({ x, y, value, label, color = mint }: any) {
  return (
    <g>
      <Text x={x} y={y} size={72} bold color={color}>
        {value}
      </Text>
      <Text x={x} y={y + 48} color={muted}>
        {label}
      </Text>
    </g>
  );
}
function Visual({ i, f, n }: { i: number; f: number; n: number }) {
  const p = ramp(f, 12, 84);
  const later = ramp(f, n * 0.43, n * 0.49);
  switch (i) {
    case 0:
      return (
        <>
          <Text x={75} y={296} size={33}>
            Um classificador estima o risco de inadimplência.
          </Text>
          {["Dados de crédito", "Modelo", "Decisão"].map((x, j) => (
            <g key={x} opacity={ramp(f, j * 17, 28 + j * 17)}>
              <rect
                x={78 + j * 403}
                y={365}
                rx="14"
                width="317"
                height="100"
                fill="#203d42"
              />
              <Text x={105 + j * 403} y={426} size={32} bold>
                {x}
              </Text>
              {j < 2 && (
                <path
                  d={`M${405 + j * 403} 416h61m-15 -13l16 13 -16 13`}
                  stroke={mint}
                  fill="none"
                  strokeWidth="4"
                />
              )}
            </g>
          ))}
          <g opacity={later}>
            <Text x={76} y={555} size={35} color={amber}>
              As respostas continuam. O desempenho pode mudar.
            </Text>
          </g>
        </>
      );
    case 1:
      return (
        <>
          <Stat x={74} y={316} value="30.000" label="registros de crédito" />
          {[
            [18000, "Referência", "Treino", mint],
            [6000, "Calibração", "Escolha do limiar", "#77a9cc"],
            [6000, "Produção", "Simulação", amber],
          ].map(([v, l, sub, c], j) => (
            <g key={l}>
              <rect
                x={[76, 728, 962][j]}
                y={418}
                width={[630, 213, 213][j] * p}
                height={36}
                rx={3}
                fill={String(c)}
              />
              <Text x={[76, 728, 962][j]} y={508} size={32} bold>
                {Number(v).toLocaleString("pt-BR")}
              </Text>
              <Text x={[76, 728, 962][j]} y={548}>
                {l}
              </Text>
              <Text x={[76, 728, 962][j]} y={588} size={22} color={muted}>
                {sub}
              </Text>
            </g>
          ))}
        </>
      );
    case 2:
      return (
        <>
          {f < n * 0.51 ? (
            <>
              <Text x={76} y={298} size={33}>
                Faixa de idade aceita: 19 a 100 anos
              </Text>
              {Array.from({ length: 32 }, (_, j) => (
                <rect
                  key={j}
                  x={82 + (j % 16) * 39}
                  y={362 + Math.floor(j / 16) * 42}
                  width={28}
                  height={28}
                  rx={4}
                  fill={f > 50 + j * 2 ? amber : "#386167"}
                />
              ))}
              <Stat
                x={800}
                y={410}
                value="32 / 32"
                label="idades fora da faixa"
                color={amber}
              />
              <Text x={76} y={565} size={38} bold>
                Lote rejeitado antes da predição
              </Text>
            </>
          ) : (
            <>
              <Text x={76} y={294} size={36}>
                Regressão logística + escalonamento na referência
              </Text>
              <Stat
                x={76}
                y={432}
                value="0,27"
                label="limiar escolhido na calibração"
              />
              <Stat
                x={737}
                y={432}
                value="0,504"
                label="F1 de desenvolvimento"
              />
              <Text x={76} y={582} color={muted}>
                O mesmo conjunto orientou o limiar e a avaliação.
              </Text>
            </>
          )}
        </>
      );
    case 3:
      return (
        <>
          <Text x={76} y={275} size={30}>
            PSI observado na produção simulada
          </Text>
          {[
            ["LIMIT_BAL", 0.382789, 0.006938, "+50% no limite"],
            ["PAY_AMT1", 0.458898, 0.008578, "−50% no pagamento"],
          ].map(([l, v, limit, op], j) => (
            <g key={l}>
              <Text x={76} y={358 + j * 98} bold>
                {l}
              </Text>
              <Text x={76} y={389 + j * 98} size={20} color={muted}>
                {op}
              </Text>
              <rect
                x={390}
                y={335 + j * 98}
                width={570}
                height={27}
                rx={3}
                fill="#28414a"
              />
              <rect
                x={390}
                y={335 + j * 98}
                width={Number(v) * 1140 * p}
                height={27}
                fill={amber}
              />
              <line
                x1={390 + Number(limit) * 1140}
                x2={390 + Number(limit) * 1140}
                y1={329 + j * 98}
                y2={367 + j * 98}
                stroke="white"
                strokeWidth="3"
              />
              <Text x={990} y={360 + j * 98} size={33}>
                {Number(v).toFixed(3).replace(".", ",")}
              </Text>
            </g>
          ))}
          <Text x={76} y={565} size={33} color={mint}>
            Controle: zero alertas. Cenário alterado: dois.
          </Text>
          <Text x={76} y={610} color={muted} size={21}>
            Traço branco: limite empírico de PSI calibrado por variável.
          </Text>
        </>
      );
    case 5:
      return (
        <>
          {f < n * 0.47 ? (
            <>
              <image
                href={staticFile("mlflow.png")}
                x="80"
                y="232"
                width="771"
                height="460"
                preserveAspectRatio="xMinYMin meet"
              />
              <Stat x={870} y={352} value="6.000" label="linhas processadas" />
              <Stat
                x={870}
                y={540}
                value="2"
                label="alertas de drift"
                color={amber}
              />
            </>
          ) : (
            <>
              <Text x={76} y={300} size={33}>
                Predição no cenário de Data Drift
              </Text>
              <Stat x={76} y={439} value="47 ms" label="tempo decorrido" />
              <Stat x={468} y={439} value="16 ms" label="tempo de CPU" />
              <Stat x={856} y={439} value="260" label="MiB de memória" />
              <Text x={76} y={593} color={muted}>
                Eventos vinculam cada alerta à regra, ao valor e à ação
                sugerida.
              </Text>
            </>
          )}
        </>
      );
    case 6:
      return (
        <>
          {f < n * 0.43 ? (
            <>
              <Text x={76} y={297} size={32}>
                Dados individuais ficam no ambiente local.
              </Text>
              <Text x={76} y={396} color={mint} size={23}>
                FORA DA MATRIZ DE PREDIÇÃO
              </Text>
              <Text x={76} y={453} size={39}>
                ID, sexo, escolaridade e estado civil
              </Text>
              <Text x={76} y={552} color={muted}>
                Idade entra no modelo e na auditoria por grupos.
              </Text>
            </>
          ) : (
            <>
              {[
                ["19–29", 12.43],
                ["30–44", 9.88],
                ["45–59", 17.43],
                ["60–100", 26.23],
              ].map(([l, v], j) => (
                <g key={l}>
                  <Text x={75} y={291 + 63 * j}>
                    {l} anos
                  </Text>
                  <rect
                    x={268}
                    y={269 + j * 63}
                    width={
                      Number(v) *
                      23 *
                      ramp(f, n * 0.42 + j * 3, n * 0.49 + j * 3)
                    }
                    height="26"
                    fill={j === 3 ? amber : mint}
                  />
                  <Text x={925} y={293 + 63 * j}>
                    {String(v).replace(".", ",")}%
                  </Text>
                </g>
              ))}
              <Text x={75} y={566} size={34}>
                Diferença de 16,35 pontos percentuais.
              </Text>
              <Text x={75} y={613} size={24} color={muted}>
                A faixa mais velha reúne apenas 82 registros.
              </Text>
            </>
          )}
        </>
      );
    default:
      return (
        <>
          <Stat x={76} y={377} value="263" label="testes aprovados no CI" />
          <Text x={600} y={315} size={29}>
            1. Conferir a origem dos dados
          </Text>
          <Text x={600} y={384} size={29}>
            2. Avaliar o impacto com rótulos
          </Text>
          <Text x={600} y={453} size={29}>
            3. Decidir sobre o retreino
          </Text>
          <g opacity={later}>
            <Text x={76} y={564} color={mint} size={24}>
              REPOSITÓRIO PÚBLICO
            </Text>
            <Text x={76} y={612} size={31}>
              github.com/primodeckers/tech-challenge-fase-four
            </Text>
          </g>
        </>
      );
  }
}
function Scene({ i, n }: { i: number; n: number }) {
  const f = useCurrentFrame();
  if (i === 4) return <ConceptDrift />;
  return (
    <AbsoluteFill
      style={{
        fontFamily: "Segoe UI, Arial, sans-serif",
        background: "linear-gradient(120deg,#10272e,#101e26 72%,#1f3439)",
      }}
    >
      <svg viewBox="0 0 1280 720" width="1280" height="720">
        <Text x={72} y={65} size={21} color={mint}>
          {labels[i]}　 /　 MONITORAMENTO DE CRÉDITO
        </Text>
        <g
          opacity={ramp(f, 0, 13)}
          transform={`translate(0 ${10 * (1 - ramp(f, 0, 13))})`}
        >
          <Text x={68} y={148} size={56} bold>
            {titles[i]}
          </Text>
          <Text x={73} y={198} size={24} color={muted}>
            {notes[i]}
          </Text>
          <Visual i={i} f={f} n={n} />
        </g>
        <rect x="72" y="693" width="1136" height="2" fill="#34515a" />
        <rect x="72" y="693" width={(1136 * f) / n} height="2" fill={mint} />
      </svg>
    </AbsoluteFill>
  );
}
function Film() {
  return (
    <AbsoluteFill style={{ background: "#10252c" }}>
      {timing.starts.map((start, i) => {
        const n = (timing.starts[i + 1] ?? timing.end) - start;
        return (
          <Sequence key={i} from={start} durationInFrames={n}>
            <Scene i={i} n={n} />
          </Sequence>
        );
      })}
      <Sequence from={12}>
        <Audio src={staticFile("narracao.mp3")} />
      </Sequence>
    </AbsoluteFill>
  );
}
registerRoot(() => (
  <Composition
    id="EntregaSTAR"
    component={Film}
    width={1280}
    height={720}
    fps={24}
    durationInFrames={timing.end}
  />
));
