const metricsSummaryEl = document.getElementById("metrics-summary");
const agentStatsEl = document.getElementById("agent-stats");
const tasksListEl = document.getElementById("tasks-list");
const taskCountEl = document.getElementById("task-count");
const blackboardTaskEl = document.getElementById("blackboard-task");
const blackboardEntriesEl = document.getElementById("blackboard-entries");
const playgroundForm = document.getElementById("playground-form");
const playgroundInput = document.getElementById("playground-input");
const playgroundStatusEl = document.getElementById("playground-status");
const playgroundOutputEl = document.getElementById("playground-output");

let selectedTaskId = null;
let resultPoller = null;

const formatter = new Intl.DateTimeFormat("en-US", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) {
    return "-";
  }
  return `${seconds.toFixed(1)}s`;
}

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function renderMetrics(data) {
  const { performance, agent_stats: agentStats, system } = data;
  metricsSummaryEl.innerHTML = `
    <div class="metric-card">
      <span>Total Queries</span>
      <strong>${performance.total_queries}</strong>
    </div>
    <div class="metric-card">
      <span>Success Rate</span>
      <strong>${(performance.success_rate * 100).toFixed(1)}%</strong>
    </div>
    <div class="metric-card">
      <span>Avg Execution</span>
      <strong>${formatDuration(performance.average_execution_time_seconds)}</strong>
    </div>
    <div class="metric-card">
      <span>Agents</span>
      <strong>${system.agents}</strong>
    </div>
  `;

  if (!agentStats || Object.keys(agentStats).length === 0) {
    agentStatsEl.innerHTML = "<p>No agent runtime data yet.</p>";
    return;
  }

  const rows = Object.entries(agentStats)
    .map(
      ([agentId, avgSeconds]) => `
        <div class="badge">
          <strong>${agentId}</strong>
          <span>${formatDuration(avgSeconds)}</span>
        </div>
      `
    )
    .join("");

  agentStatsEl.innerHTML = `
    <h3>Average Agent Duration</h3>
    <div class="agent-badges">${rows}</div>
  `;
}

function renderTasks(tasks) {
  taskCountEl.textContent = `${tasks.length} total`;
  tasksListEl.innerHTML = "";

  if (!selectedTaskId && tasks.length > 0) {
    selectedTaskId = tasks[0].task_id;
  }

  tasks.forEach((task) => {
    const card = document.createElement("div");
    card.className = "task-card";
    if (task.task_id === selectedTaskId) {
      card.classList.add("active");
    }

    const updatedAt = formatter.format(new Date(task.updated_at));
    const activeAgents = (task.activity && task.activity.active_agents) || [];

    card.innerHTML = `
      <div class="task-status">${task.status}</div>
      <div class="task-meta">Updated ${updatedAt}</div>
      <div class="task-meta">Entries: ${task.total_entries}</div>
      <div class="task-meta">Active: ${activeAgents.length}</div>
    `;

    card.addEventListener("click", () => {
      selectedTaskId = task.task_id;
      renderTasks(tasks);
      loadBlackboard();
    });

    tasksListEl.appendChild(card);
  });
}

function renderBlackboard(data) {
  if (!data || !data.entries) {
    blackboardTaskEl.textContent = "No task selected";
    blackboardEntriesEl.innerHTML = "";
    return;
  }

  blackboardTaskEl.textContent = data.task_id;
  blackboardEntriesEl.innerHTML = "";

  data.entries.forEach((entry) => {
    const card = document.createElement("div");
    card.className = "blackboard-entry";
    card.innerHTML = `
      <div class="entry-header">
        <span>${entry.entry_type}</span>
        <span>${formatter.format(new Date(entry.timestamp))}</span>
      </div>
      <p><strong>${entry.agent_id}</strong>: ${entry.content_preview}</p>
    `;
    blackboardEntriesEl.appendChild(card);
  });
}

function renderPlaygroundResult(result) {
  playgroundOutputEl.innerHTML = "";
  if (!result) {
    playgroundOutputEl.innerHTML = "<p>No response yet.</p>";
    return;
  }

  const header = document.createElement("div");
  header.className = "task-meta";
  header.textContent = `Task ${result.task_id} • ${result.success ? "Success" : "Error"}`;
  playgroundOutputEl.appendChild(header);

  const meta = document.createElement("div");
  meta.className = "task-meta";
  meta.textContent = `Execution: ${formatDuration(result.execution_time_seconds || 0)}`;
  playgroundOutputEl.appendChild(meta);

  const pre = document.createElement("pre");
  pre.textContent = result.success ? (result.answer || "") : (result.error || "Unknown error");
  playgroundOutputEl.appendChild(pre);
}

async function loadMetrics() {
  try {
    const data = await fetchJSON("/ui/metrics");
    renderMetrics(data);
  } catch (error) {
    metricsSummaryEl.innerHTML = `<p class="error">Failed to load metrics.</p>`;
    console.error(error);
  }
}

async function loadTasks() {
  try {
    const data = await fetchJSON("/ui/tasks");
    renderTasks(data.tasks || []);
    await loadBlackboard();
  } catch (error) {
    tasksListEl.innerHTML = `<p class="error">Failed to load tasks.</p>`;
    console.error(error);
  }
}

async function loadBlackboard() {
  if (!selectedTaskId) {
    renderBlackboard(null);
    return;
  }
  try {
    const data = await fetchJSON(`/ui/task/${selectedTaskId}/blackboard?limit=20`);
    renderBlackboard(data);
  } catch (error) {
    blackboardEntriesEl.innerHTML = `<p class="error">Failed to load blackboard entries.</p>`;
    console.error(error);
  }
}

async function loadTaskResult() {
  if (!selectedTaskId) {
    return;
  }
  try {
    const response = await fetch(`/ui/task/${selectedTaskId}/result`);
    if (response.status === 404) {
      playgroundStatusEl.textContent = "Running...";
      playgroundStatusEl.classList.remove("error");
      return;
    }
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
  const data = await response.json();
  playgroundStatusEl.classList.remove("error");
    renderPlaygroundResult(data);
    playgroundStatusEl.textContent = data.success ? "Completed" : "Failed";
    if (resultPoller && data.success !== undefined) {
      clearInterval(resultPoller);
      resultPoller = null;
    }
    await loadTasks();
    await loadBlackboard();
  } catch (error) {
    console.error(error);
    playgroundStatusEl.textContent = "Result fetch failed";
    playgroundStatusEl.classList.add("error");
  }
}

async function submitPlaygroundQuery(event) {
  event.preventDefault();
  const query = playgroundInput.value.trim();
  if (!query) {
    playgroundStatusEl.textContent = "Enter a prompt first.";
    return;
  }

  playgroundStatusEl.textContent = "Sending...";
  playgroundStatusEl.classList.remove("error");

  try {
    const response = await fetch("/ui/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    playgroundStatusEl.textContent = "Running...";
    renderPlaygroundResult(null);
    selectedTaskId = data.task_id;
    playgroundInput.value = "";
    await loadTasks();
    await loadTaskResult();
    if (resultPoller) {
      clearInterval(resultPoller);
    }
    resultPoller = setInterval(loadTaskResult, 4000);
  } catch (error) {
    console.error(error);
    playgroundStatusEl.textContent = "Request failed";
    playgroundStatusEl.classList.add("error");
  }
}

function startPolling() {
  loadMetrics();
  loadTasks();
  setInterval(loadMetrics, 5000);
  setInterval(loadTasks, 5000);
}

window.addEventListener("DOMContentLoaded", startPolling);
if (playgroundForm) {
  playgroundForm.addEventListener("submit", submitPlaygroundQuery);
  renderPlaygroundResult(null);
}
