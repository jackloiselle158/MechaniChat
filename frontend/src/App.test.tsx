import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import App from "./App";

describe("MechaniChat frontend", () => {
  beforeEach(() => localStorage.clear());
  afterEach(cleanup);

  it("renders the new-chat experience and prominent photo control", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: /clear answers for.*the road ahead/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /add photos/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /start a new conversation/i })).toBeInTheDocument();
  });

  it("opens both vehicle entry methods", () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /add your vehicle/i }));
    expect(screen.getByRole("dialog", { name: /your vehicles/i })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "VIN" }));
    expect(screen.getByLabelText(/vehicle identification number/i)).toBeInTheDocument();
  });

  it("selects a saved vehicle for a new conversation", () => {
    localStorage.setItem("mechanichat.app-data.v1", JSON.stringify({
      conversations: [{
        id: "existing-conversation",
        title: "Existing conversation",
        createdAt: 1,
        updatedAt: 1,
        messages: [],
        vehicleId: "daily-vehicle",
      }],
      vehicles: [{
        id: "daily-vehicle",
        nickname: "Daily",
        input_mode: "details",
        year: "2011",
        make: "GMC",
        model: "Terrain",
      }],
      activeConversationId: "existing-conversation",
    }));
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /start a new conversation/i }));
    expect(screen.getByRole("button", { name: /add your vehicle/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /add your vehicle/i }));
    fireEvent.click(screen.getByRole("button", { name: /use daily for this conversation/i }));

    expect(screen.getByRole("button", { name: /manage active vehicle daily/i })).toBeInTheDocument();
  });

  it("edits a saved vehicle without creating a duplicate", async () => {
    localStorage.setItem("mechanichat.app-data.v1", JSON.stringify({
      conversations: [{
        id: "existing-conversation",
        title: "Existing conversation",
        createdAt: 1,
        updatedAt: 1,
        messages: [],
        vehicleId: "daily-vehicle",
      }],
      vehicles: [{
        id: "daily-vehicle",
        nickname: "Daily",
        input_mode: "details",
        year: "2011",
        make: "GMC",
        model: "Terrain",
      }],
      activeConversationId: "existing-conversation",
    }));
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /manage active vehicle daily.*2011 gmc terrain/i }));
    fireEvent.click(screen.getByRole("button", { name: "Edit Daily" }));
    fireEvent.change(screen.getByLabelText(/nickname/i), { target: { value: "Family car" } });
    fireEvent.click(screen.getByRole("button", { name: /update vehicle/i }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /manage active vehicle family car.*2011 gmc terrain/i })).toBeInTheDocument();
    });
    const savedData = JSON.parse(localStorage.getItem("mechanichat.app-data.v1") ?? "{}");
    expect(savedData.vehicles).toHaveLength(1);
    expect(savedData.vehicles[0]).toMatchObject({
      id: "daily-vehicle",
      nickname: "Family car",
    });
  });

  it("shows decoded vehicle details instead of a VIN", () => {
    localStorage.setItem("mechanichat.app-data.v1", JSON.stringify({
      conversations: [{
        id: "vin-conversation",
        title: "VIN conversation",
        createdAt: 1,
        updatedAt: 1,
        messages: [],
        vehicleId: "vin-vehicle",
      }],
      vehicles: [{
        id: "vin-vehicle",
        nickname: "Dad's car",
        input_mode: "vin",
        vin: "1HGCM82633A004352",
        year: "2003",
        make: "HONDA",
        model: "Accord",
      }],
      activeConversationId: "vin-conversation",
    }));
    render(<App />);

    expect(screen.getByRole("button", { name: /manage active vehicle dad's car.*2003 honda accord/i })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /manage active vehicle dad's car.*2003 honda accord/i }));
    expect(screen.getAllByText("2003 HONDA Accord").length).toBeGreaterThanOrEqual(2);
    expect(screen.queryByText(/4352/)).not.toBeInTheDocument();
  });

  it("shows a concise request summary and the conversation vehicle", () => {
    const firstRequest = "Could you please explain why my brakes squeal when I stop?";
    localStorage.setItem("mechanichat.app-data.v1", JSON.stringify({
      conversations: [{
        id: "brake-conversation",
        title: firstRequest.slice(0, 42),
        createdAt: 1,
        updatedAt: 1,
        messages: [{
          id: "first-message",
          role: "user",
          content: firstRequest,
          createdAt: 1,
        }],
        vehicleId: "daily-vehicle",
      }],
      vehicles: [{
        id: "daily-vehicle",
        nickname: "Daily",
        input_mode: "details",
        year: "2011",
        make: "GMC",
        model: "Terrain",
      }],
      activeConversationId: "brake-conversation",
    }));
    render(<App />);

    expect(screen.getByText("Explain why my brakes squeal when I stop?")).toBeInTheDocument();
    expect(screen.getByText(/Daily · 2011 GMC Terrain/)).toBeInTheDocument();
  });

  it("grows the message input before enabling its scrollbar", async () => {
    render(<App />);
    const messageInput = screen.getByLabelText("Message MechaniChat") as HTMLTextAreaElement;
    let contentHeight = 60;
    Object.defineProperty(messageInput, "scrollHeight", {
      configurable: true,
      get: () => contentHeight,
    });

    fireEvent.change(messageInput, { target: { value: "A short message" } });
    await waitFor(() => expect(messageInput.style.overflowY).toBe("hidden"));
    expect(Number.parseFloat(messageInput.style.height)).toBe(60);

    contentHeight = 400;
    fireEvent.change(messageInput, {
      target: { value: "A long message\n".repeat(12) },
    });
    await waitFor(() => expect(messageInput.style.overflowY).toBe("auto"));
    expect(Number.parseFloat(messageInput.style.height)).toBeLessThanOrEqual(177);
  });
});
