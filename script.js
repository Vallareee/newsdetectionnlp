document.getElementById("checkBtn").addEventListener("click", checkNews);

async function checkNews() {
  const inputText = document.getElementById("newsInput").value.trim();
  const resultDiv = document.getElementById("result");

  if (!inputText) {
    resultDiv.textContent = "⚠️ Please enter some text.";
    resultDiv.className = "result";
    return;
  }

  // backend integration
  
  // const data = await response.json();
  // const label = data.label; // "True", "False", or "Misleading"

  // Mock demo logic (random result for now)
  const outputs = ["True", "False", "Misleading"];
  const random = outputs[Math.floor(Math.random() * outputs.length)];
  showResult(random);
}

function showResult(label) {
  const resultDiv = document.getElementById("result");
  resultDiv.textContent = "Result: " + label;

  // Reset classes
  resultDiv.className = "result";

  // Add color-coded class
  if (label === "True") {
    resultDiv.classList.add("true");
  } else if (label === "False") {
    resultDiv.classList.add("false");
  } else if (label === "Misleading") {
    resultDiv.classList.add("misleading");
  }
}
