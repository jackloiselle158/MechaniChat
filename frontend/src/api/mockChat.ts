import type { Source, Vehicle } from "../types";

export type StreamEvent =
  | { type: "delta"; text: string }
  | { type: "sources"; sources: Source[] }
  | { type: "complete" };

const sampleSources: Source[] = [
  {
    id: 1,
    title: "NHTSA — Vehicle safety guidance",
    detail: "General vehicle safety and recall information",
    url: "https://www.nhtsa.gov/",
  },
  {
    id: 2,
    title: "MechaniChat service library",
    detail: "Brake-system inspection and symptom reference",
  },
];

function buildResponse(prompt: string, vehicle?: Vehicle): string {
  const vehicleName = vehicle
    ? vehicle.input_mode === "vin"
      ? `the vehicle with VIN ending in ${vehicle.vin?.slice(-4)}`
      : `your ${vehicle.year} ${vehicle.make} ${vehicle.model}`
    : "your vehicle";

  return `Based on what you described about ${vehicleName}, I’d start with three likely causes:\n\n1. **Brake-pad wear** — a high-pitched squeal often comes from the wear indicator touching the rotor.\n2. **Surface rust or debris** — light corrosion can make noise after rain or after the car has been parked.\n3. **A sticking caliper** — especially if the car pulls to one side or one wheel feels unusually hot.\n\n**What to check:** Note whether the sound changes with pedal pressure, look through the wheel for deep rotor grooves, and avoid driving if braking feels weak or the pedal vibrates heavily. Because brakes are safety-critical, arrange an in-person inspection if the sound continues. [1][2]\n\n_This is a preview response from the frontend mock. Your message was: “${prompt}”_`;
}

export async function* streamMockResponse(
  prompt: string,
  vehicle: Vehicle | undefined,
  signal: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const response = buildResponse(prompt, vehicle);
  const chunks = response.match(/.{1,18}(?:\s|$)/g) ?? [response];

  for (const chunk of chunks) {
    if (signal.aborted) throw new DOMException("Request aborted", "AbortError");
    await new Promise((resolve) => window.setTimeout(resolve, 28));
    yield { type: "delta", text: chunk };
  }

  yield { type: "sources", sources: sampleSources };
  yield { type: "complete" };
}
