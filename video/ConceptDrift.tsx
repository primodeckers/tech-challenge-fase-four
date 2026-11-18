import React from "react";
import {
  AbsoluteFill,
  Composition,
  Easing,
  interpolate,
  registerRoot,
  useCurrentFrame,
} from "remotion";

const ink = "#f2f5ed",
  muted = "#adbdbe",
  mint = "#69dfc5",
  peach = "#ffa56f";
const clamped = {
  extrapolateLeft: "clamp",
  extrapolateRight: "clamp",
} as const;
const title = {
  fontFamily: "Segoe UI, Arial, sans-serif",
  fontWeight: 650,
  letterSpacing: -2,
};

export function ConceptDrift() {
  const frame = useCurrentFrame();
  const phase = interpolate(frame, [58, 157], [0, 1], {
    ...clamped,
    easing: Easing.inOut(Easing.cubic),
  });
  const swapped = Math.round(174 * phase);
  const reveal = interpolate(frame, [154, 184], [0, 1], clamped);
  return (
    <AbsoluteFill
      style={{
        background: "linear-gradient(135deg,#10252c,#102027 65%,#182c31)",
        color: ink,
        fontFamily: "Segoe UI,Arial,sans-serif",
      }}
    >
      <svg
        width="1280"
        height="720"
        viewBox="0 0 1280 720"
        role="img"
        aria-label="Mudança de rótulo em dois grupos mantendo as entradas fixas"
      >
        <text x="72" y="72" fill={mint} fontSize="21" letterSpacing="3">
          CONCEPT DRIFT / SIMULAÇÃO
        </text>
        <text x="68" y="155" fill={ink} fontSize="67" style={title}>
          Mesmas entradas, outros rótulos
        </text>
        <text x="72" y="203" fill={muted} fontSize="28">
          Duas trocas equilibradas preservam o total de inadimplentes.
        </text>
        {[0, 1].map((g) => {
          const x = 72 + g * 584;
          const src = g === 0 ? mint : peach;
          const dst = g === 0 ? peach : mint;
          return (
            <g key={g} transform={`translate(${x},255)`}>
              <rect
                width="552"
                height="232"
                rx="19"
                fill="#182e36"
                stroke="#38505a"
              />
              <text x="25" y="44" fill={ink} fontSize="25" fontWeight="600">
                {g === 0 ? "PAY_0 ≤ 0" : "PAY_0 > 0"}
              </text>
              <text x="525" y="44" textAnchor="end" fill={muted} fontSize="23">
                {g === 0 ? "Rótulo 0 → 1" : "Rótulo 1 → 0"}
              </text>
              {Array.from({ length: 174 }, (_, i) => {
                const cx = 29 + (i % 29) * 17.5,
                  cy = 77 + Math.floor(i / 29) * 17;
                const changed = i < swapped;
                const isOne = g === 0 ? changed : !changed;
                return isOne ? (
                  <rect
                    key={i}
                    x={cx - 5}
                    y={cy - 5}
                    width="10"
                    height="10"
                    rx="1"
                    transform={`rotate(45 ${cx} ${cy})`}
                    fill={changed ? dst : src}
                  />
                ) : (
                  <circle
                    key={i}
                    cx={cx}
                    cy={cy}
                    r="5.4"
                    fill={changed ? dst : src}
                  />
                );
              })}
              <text x="25" y="205" fill={ink} fontSize="23">
                {swapped} de 174 rótulos alterados
              </text>
            </g>
          );
        })}
        <g opacity={reveal}>
          <path d="M73 540H1207" stroke="#436169" />
          <text x="72" y="599" fill={ink} fontSize="42" style={title}>
            AUC
          </text>
          <text x="186" y="599" fill={mint} fontSize="43" fontWeight="650">
            0,723
          </text>
          <text x="323" y="599" fill={muted} fontSize="42">
            →
          </text>
          <text x="393" y="599" fill={peach} fontSize="43" fontWeight="650">
            0,630
          </text>
          <text x="636" y="594" fill={ink} fontSize="28">
            348 rótulos alterados em 6.000 registros
          </text>
        </g>
        <text x="72" y="662" fill={muted} fontSize="21">
          Cada marca representa um registro alterado na simulação. As posições
          são ilustrativas.
        </text>
      </svg>
    </AbsoluteFill>
  );
}
