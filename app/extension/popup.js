async function main() {
    const root = document.getElementById("root");
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      const url = tab?.url || "";
      if (!url || url.startsWith("chrome://")) {
        root.textContent = "No readable page URL.";
        return;
      }
      root.textContent = "Summarizing…";
      const res = await fetch(
        "http://127.0.0.1:8000/analyze?url=" + encodeURIComponent(url)
      );
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || res.statusText);
      }
      const data = await res.json();
      root.innerHTML = `
        <div style="font-weight:600; margin-bottom:6px;">${data.title || ""}</div>
        <div style="margin:6px 0;"><b>Answer:</b> ${data.answer || ""}</div>
        <ul style="padding-left:18px; margin:6px 0;">
          ${(data.summary || []).map(s => `<li>${s}</li>`).join("")}
        </ul>
      `;
    } catch (e) {
      root.textContent = "Error: " + (e?.message || e);
    }
  }
  main();
  