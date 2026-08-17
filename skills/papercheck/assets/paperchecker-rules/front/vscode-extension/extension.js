const vscode = require('vscode');
const path = require('path');
const { spawn } = require('child_process');
const { URL } = require('url');

let serverProcess = null;
let outputChannel;

function getWorkspaceRoot() {
    const folders = vscode.workspace.workspaceFolders;
    if (!folders || folders.length === 0) {
        vscode.window.showErrorMessage('请先打开包含 PaperChecker 项目的工作区。');
        return undefined;
    }
    return folders[0].uri.fsPath;
}

function getConfiguration() {
    return vscode.workspace.getConfiguration('paperchecker');
}

function getPythonCommand() {
    const config = getConfiguration();
    return config.get('pythonPath') || 'python';
}

function getServerUrl() {
    const config = getConfiguration();
    return config.get('serverUrl') || 'http://127.0.0.1:8000';
}

class PaperCheckerTreeDataProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        this.serverStatus = 'stopped'; // 'stopped', 'starting', 'running'
    }

    refresh() {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element) {
        return element;
    }

    getChildren(element) {
        if (element) {
            return [];
        }

        const items = [];

        // Server status item
        const serverItem = new vscode.TreeItem(
            this.serverStatus === 'running' ? '后端服务: 运行中' : '后端服务: 未运行',
            vscode.TreeItemCollapsibleState.None
        );

        if (this.serverStatus === 'running') {
            serverItem.description = 'http://127.0.0.1:8000';
            serverItem.iconPath = new vscode.ThemeIcon('server-process');
            serverItem.contextValue = 'server-running';
        } else {
            serverItem.iconPath = new vscode.ThemeIcon('server-environment');
            serverItem.contextValue = 'server-not-running';
        }

        items.push(serverItem);

        // Start Server button
        const startItem = new vscode.TreeItem('启动后端', vscode.TreeItemCollapsibleState.None);
        startItem.iconPath = new vscode.ThemeIcon('play');
        startItem.command = {
            command: 'paperchecker.startServer',
            title: '启动后端',
            arguments: []
        };
        items.push(startItem);

        // Stop Server button
        if (this.serverStatus === 'running') {
            const stopItem = new vscode.TreeItem('停止后端', vscode.TreeItemCollapsibleState.None);
            stopItem.iconPath = new vscode.ThemeIcon('stop');
            stopItem.command = {
                command: 'paperchecker.stopServer',
                title: '停止后端',
                arguments: []
            };
            items.push(stopItem);
        }

        // Analyze Document button
        const analyzeItem = new vscode.TreeItem('分析报告', vscode.TreeItemCollapsibleState.None);
        analyzeItem.iconPath = new vscode.ThemeIcon('file-code');
        analyzeItem.command = {
            command: 'paperchecker.analyzeDocument',
            title: '分析报告',
            arguments: []
        };
        items.push(analyzeItem);

        // Open Web Interface button
        const webItem = new vscode.TreeItem('打开网页界面', vscode.TreeItemCollapsibleState.None);
        webItem.iconPath = new vscode.ThemeIcon('globe');
        webItem.command = {
            command: 'paperchecker.openWebInterface',
            title: '打开网页界面',
            arguments: []
        };
        items.push(webItem);

        return items;
    }
}

// Global reference to the tree data provider to update server status
let treeDataProvider = null;

function activate(context) {
    outputChannel = vscode.window.createOutputChannel('PaperChecker');
    context.subscriptions.push(outputChannel);

    // Create tree view data provider
    treeDataProvider = new PaperCheckerTreeDataProvider();

    // Register the tree view
    const treeView = vscode.window.createTreeView('paperchecker.explorer', {
        treeDataProvider: treeDataProvider
    });

    // Update tree view when server status changes
    function updateTree() {
        treeDataProvider.refresh();
    }

    const subscriptions = [
        vscode.commands.registerCommand('paperchecker.startServer', startServer),
        vscode.commands.registerCommand('paperchecker.stopServer', stopServer),
        vscode.commands.registerCommand('paperchecker.analyzeDocument', analyzeDocument),
        vscode.commands.registerCommand('paperchecker.convertJsonToMarkdown', convertJsonToMarkdown),
        vscode.commands.registerCommand('paperchecker.openWebInterface', openWebInterface),
    ];

    subscriptions.forEach((item) => context.subscriptions.push(item));
}

function deactivate() {
    stopServer();
}

function getWorkspaceRoot() {
    const folders = vscode.workspace.workspaceFolders;
    if (!folders || folders.length === 0) {
        vscode.window.showErrorMessage('请先打开包含 PaperChecker 项目的工作区。');
        return undefined;
    }
    return folders[0].uri.fsPath;
}

function getConfiguration() {
    return vscode.workspace.getConfiguration('paperchecker');
}

function getPythonCommand() {
    const config = getConfiguration();
    return config.get('pythonPath') || 'python';
}

function getServerUrl() {
    const config = getConfiguration();
    return config.get('serverUrl') || 'http://127.0.0.1:8000';
}

async function startServer() {
    if (serverProcess) {
        vscode.window.showInformationMessage('PaperChecker 后端已在运行。');
        return true;
    }

    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        return false;
    }

    const pythonCmd = getPythonCommand();
    const scriptPath = path.join(workspaceRoot, 'run_server.py');

    outputChannel.appendLine(`[PaperChecker] 使用 ${pythonCmd} 启动服务器：${scriptPath}`);

    serverProcess = spawn(pythonCmd, [scriptPath], {
        cwd: workspaceRoot,
        shell: process.platform === 'win32',
    });

    serverProcess.stdout.on('data', (data) => outputChannel.append(data.toString()));
    serverProcess.stderr.on('data', (data) => outputChannel.append(data.toString()));
    serverProcess.on('exit', (code) => {
        outputChannel.appendLine(`[PaperChecker] 服务器进程退出，代码：${code}`);
        serverProcess = null;
    });

    vscode.window.showInformationMessage('PaperChecker 后端服务启动中，详情见输出面板。');
    return true;
}

function stopServer() {
    if (!serverProcess) {
        vscode.window.showInformationMessage('PaperChecker 后端未在运行。');
        return;
    }

    serverProcess.kill();
    serverProcess = null;
    vscode.window.showInformationMessage('已停止 PaperChecker 后端服务。');
}

async function ensureServerReachable(serverUrl) {
    const healthUrl = serverUrl.replace(/\/$/, '') + '/api/health';
    try {
        const response = await fetch(healthUrl, { method: 'GET' });
        if (response.ok) {
            return true;
        }
    } catch (error) {
        outputChannel.appendLine(`[PaperChecker] 健康检查失败：${error}`);
    }

    const choice = await vscode.window.showWarningMessage(
        '无法连接到 PaperChecker 后端，是否尝试启动本地服务？',
        '启动服务',
        '取消'
    );

    if (choice === '启动服务') {
        const started = await startServer();
        if (!started) {
            return false;
        }
        for (let attempt = 0; attempt < 15; attempt += 1) {
            await delay(1000);
            try {
                const response = await fetch(healthUrl, { method: 'GET' });
                if (response.ok) {
                    return true;
                }
            } catch (error) {
                outputChannel.appendLine(`[PaperChecker] 重试健康检查失败：${error}`);
            }
        }
        vscode.window.showErrorMessage('后端启动失败，请查看输出面板日志。');
        return false;
    }

    return false;
}

async function analyzeDocument() {
    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        return;
    }

    const serverUrl = getServerUrl();
    const reachable = await ensureServerReachable(serverUrl);
    if (!reachable) {
        return;
    }

    const docUris = await vscode.window.showOpenDialog({
        filters: { Word: ['docx', 'doc'] },
        canSelectMany: false,
        openLabel: '选择待分析文档',
    });
    if (!docUris || docUris.length === 0) {
        return;
    }

    const documentPath = docUris[0].fsPath;
    const config = getConfiguration();
    const bridgeScript = path.join(workspaceRoot, 'utils', 'vscode_bridge.py');
    const args = [
        '--document',
        documentPath,
        '--server',
        serverUrl,
        '--workspace',
        workspaceRoot,
        '--json-dir',
        config.get('jsonOutputDir'),
        '--md-dir',
        config.get('markdownOutputDir'),
    ];

    try {
        const result = await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'PaperChecker 正在分析文档...',
                cancellable: false,
            },
            async () => runPythonScript(bridgeScript, args)
        );

        const payload = parseJsonOutput(result.stdout);
        vscode.window.showInformationMessage(`报告生成成功：${payload.markdown_path}`);
        outputChannel.appendLine(`[PaperChecker] JSON: ${payload.json_path}`);
        outputChannel.appendLine(`[PaperChecker] Markdown: ${payload.markdown_path}`);
    } catch (error) {
        vscode.window.showErrorMessage(`生成报告失败：${error.message}`);
    }
}

async function convertJsonToMarkdown() {
    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        return;
    }

    const jsonUris = await vscode.window.showOpenDialog({
        filters: { JSON: ['json'] },
        canSelectMany: false,
        openLabel: '选择 JSON 报告',
    });
    if (!jsonUris || jsonUris.length === 0) {
        return;
    }

    const jsonPath = jsonUris[0].fsPath;
    const config = getConfiguration();
    const scriptPath = path.join(workspaceRoot, 'utils', 'json_to_markdown.py');
    const args = [
        '--json',
        jsonPath,
        '--workspace',
        workspaceRoot,
        '--output-dir',
        config.get('markdownOutputDir'),
    ];

    try {
        const result = await runPythonScript(scriptPath, args);
        const payload = parseJsonOutput(result.stdout);
        vscode.window.showInformationMessage(`Markdown 已生成：${payload.markdown_path}`);
    } catch (error) {
        vscode.window.showErrorMessage(`转换失败：${error.message}`);
    }
}

function runPythonScript(scriptPath, args) {
    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        return Promise.reject(new Error('未找到工作区'));
    }

    const pythonCmd = getPythonCommand();
    return new Promise((resolve, reject) => {
        const child = spawn(pythonCmd, [scriptPath, ...args], {
            cwd: workspaceRoot,
            shell: process.platform === 'win32',
        });

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => {
            stdout += data.toString();
        });
        child.stderr.on('data', (data) => {
            stderr += data.toString();
        });
        child.on('error', (error) => {
            reject(error);
        });
        child.on('close', (code) => {
            if (code === 0) {
                resolve({ stdout, stderr });
            } else {
                reject(new Error(stderr.trim() || `Python 退出码 ${code}`));
            }
        });
    });
}

function parseJsonOutput(raw) {
    const text = raw.trim().split(/\r?\n/).filter((line) => line.trim()).pop();
    if (!text) {
        throw new Error('未收到脚本输出');
    }
    return JSON.parse(text);
}

function startServer() {
    if (serverProcess) {
        vscode.window.showInformationMessage('PaperChecker 后端已在运行。');
        if (treeDataProvider) {
            treeDataProvider.serverStatus = 'running';
            treeDataProvider.refresh();
        }
        return true;
    }

    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        return false;
    }

    const pythonCmd = getPythonCommand();
    const scriptPath = path.join(workspaceRoot, 'run_server.py');

    outputChannel.appendLine(`[PaperChecker] 使用 ${pythonCmd} 启动服务器：${scriptPath}`);

    serverProcess = spawn(pythonCmd, [scriptPath], {
        cwd: workspaceRoot,
        shell: process.platform === 'win32',
    });

    serverProcess.stdout.on('data', (data) => outputChannel.append(data.toString()));
    serverProcess.stderr.on('data', (data) => outputChannel.append(data.toString()));
    serverProcess.on('exit', (code) => {
        outputChannel.appendLine(`[PaperChecker] 服务器进程退出，代码：${code}`);
        serverProcess = null;
        if (treeDataProvider) {
            treeDataProvider.serverStatus = 'stopped';
            treeDataProvider.refresh();
        }
    });

    if (treeDataProvider) {
        treeDataProvider.serverStatus = 'starting';
        treeDataProvider.refresh();
    }

    // Wait a bit and then update status to running if server is still alive
    setTimeout(() => {
        if (serverProcess && treeDataProvider) {
            treeDataProvider.serverStatus = 'running';
            treeDataProvider.refresh();
        }
    }, 2000);

    vscode.window.showInformationMessage('PaperChecker 后端服务启动中，详情见输出面板。');
    return true;
}

function stopServer() {
    if (!serverProcess) {
        vscode.window.showInformationMessage('PaperChecker 后端未在运行。');
        if (treeDataProvider) {
            treeDataProvider.serverStatus = 'stopped';
            treeDataProvider.refresh();
        }
        return;
    }

    serverProcess.kill();
    serverProcess = null;
    if (treeDataProvider) {
        treeDataProvider.serverStatus = 'stopped';
        treeDataProvider.refresh();
    }
    vscode.window.showInformationMessage('已停止 PaperChecker 后端服务。');
}

function openWebInterface() {
    // Open the web interface in the default browser
    const serverUrl = getServerUrl();
    vscode.env.openExternal(vscode.Uri.parse(serverUrl));
}

async function ensureServerReachable(serverUrl) {
    const healthUrl = serverUrl.replace(/\/$/, '') + '/api/health';
    try {
        const response = await fetch(healthUrl, { method: 'GET' });
        if (response.ok) {
            return true;
        }
    } catch (error) {
        outputChannel.appendLine(`[PaperChecker] 健康检查失败：${error}`);
    }

    const choice = await vscode.window.showWarningMessage(
        '无法连接到 PaperChecker 后端，是否尝试启动本地服务？',
        '启动服务',
        '取消'
    );

    if (choice === '启动服务') {
        const started = await startServer();
        if (!started) {
            return false;
        }
        for (let attempt = 0; attempt < 15; attempt += 1) {
            await delay(1000);
            try {
                const response = await fetch(healthUrl, { method: 'GET' });
                if (response.ok) {
                    return true;
                }
            } catch (error) {
                outputChannel.appendLine(`[PaperChecker] 重试健康检查失败：${error}`);
            }
        }
        vscode.window.showErrorMessage('后端启动失败，请查看输出面板日志。');
        return false;
    }

    return false;
}

async function analyzeDocument() {
    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        return;
    }

    const serverUrl = getServerUrl();
    const reachable = await ensureServerReachable(serverUrl);
    if (!reachable) {
        return;
    }

    const docUris = await vscode.window.showOpenDialog({
        filters: { Word: ['docx', 'doc'] },
        canSelectMany: false,
        openLabel: '选择待分析文档',
    });
    if (!docUris || docUris.length === 0) {
        return;
    }

    const documentPath = docUris[0].fsPath;
    const config = getConfiguration();
    const bridgeScript = path.join(workspaceRoot, 'utils', 'vscode_bridge.py');
    const args = [
        '--document',
        documentPath,
        '--server',
        serverUrl,
        '--workspace',
        workspaceRoot,
        '--json-dir',
        config.get('jsonOutputDir'),
        '--md-dir',
        config.get('markdownOutputDir'),
    ];

    try {
        const result = await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'PaperChecker 正在分析文档...',
                cancellable: false,
            },
            async () => runPythonScript(bridgeScript, args)
        );

        const payload = parseJsonOutput(result.stdout);
        vscode.window.showInformationMessage(`报告生成成功：${payload.markdown_path}`);
        outputChannel.appendLine(`[PaperChecker] JSON: ${payload.json_path}`);
        outputChannel.appendLine(`[PaperChecker] Markdown: ${payload.markdown_path}`);
    } catch (error) {
        vscode.window.showErrorMessage(`生成报告失败：${error.message}`);
    }
}

async function convertJsonToMarkdown() {
    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        return;
    }

    const jsonUris = await vscode.window.showOpenDialog({
        filters: { JSON: ['json'] },
        canSelectMany: false,
        openLabel: '选择 JSON 报告',
    });
    if (!jsonUris || jsonUris.length === 0) {
        return;
    }

    const jsonPath = jsonUris[0].fsPath;
    const config = getConfiguration();
    const scriptPath = path.join(workspaceRoot, 'utils', 'json_to_markdown.py');
    const args = [
        '--json',
        jsonPath,
        '--workspace',
        workspaceRoot,
        '--output-dir',
        config.get('markdownOutputDir'),
    ];

    try {
        const result = await runPythonScript(scriptPath, args);
        const payload = parseJsonOutput(result.stdout);
        vscode.window.showInformationMessage(`Markdown 已生成：${payload.markdown_path}`);
    } catch (error) {
        vscode.window.showErrorMessage(`转换失败：${error.message}`);
    }
}

function runPythonScript(scriptPath, args) {
    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        return Promise.reject(new Error('未找到工作区'));
    }

    const pythonCmd = getPythonCommand();
    return new Promise((resolve, reject) => {
        const child = spawn(pythonCmd, [scriptPath, ...args], {
            cwd: workspaceRoot,
            shell: process.platform === 'win32',
        });

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => {
            stdout += data.toString();
        });
        child.stderr.on('data', (data) => {
            stderr += data.toString();
        });
        child.on('error', (error) => {
            reject(error);
        });
        child.on('close', (code) => {
            if (code === 0) {
                resolve({ stdout, stderr });
            } else {
                reject(new Error(stderr.trim() || `Python 退出码 ${code}`));
            }
        });
    });
}

function parseJsonOutput(raw) {
    const text = raw.trim().split(/\r?\n/).filter((line) => line.trim()).pop();
    if (!text) {
        throw new Error('未收到脚本输出');
    }
    return JSON.parse(text);
}

function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function deactivate() {
    if (serverProcess) {
        serverProcess.kill();
        serverProcess = null;
    }
}

module.exports = {
    activate,
    deactivate,
};

