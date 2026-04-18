param(
    [switch]$IncludeRuntimeArtifacts,
    [switch]$StageEverything
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($StageEverything) {
    git add -A
    git status --short
    exit $LASTEXITCODE
}

$sourceFiles = @(
    "agents/llm_provider.py",
    "api/routes.py",
    "document_summarizer/brain/brain.py",
    "document_summarizer/brain/embeddings/embedder.py",
    "document_summarizer/brain/embeddings/vector_store.py",
    "document_summarizer/brain/persistence/paper_store.py",
    "document_summarizer/brain/persistence/qa_store.py",
    "document_summarizer/server.py",
    "document_summarizer/brain/llm/provider_llm.py",
    "server.py",
    "DEPLOY_SPLIT_RENDER.md"
)

$runtimeFiles = @(
    "brain.db",
    "generated_image.png",
    "logs/app.log",
    "uploads/requirements.txt",
    "vector_indexes/paper_4.index"
)

$pycachePaths = @(
    "__pycache__/",
    "agents/__pycache__/",
    "api/__pycache__/",
    "document_summarizer/__pycache__/",
    "document_summarizer/brain/__pycache__/",
    "document_summarizer/brain/embeddings/__pycache__/",
    "document_summarizer/brain/persistence/__pycache__/",
    "document_summarizer/brain/llm/__pycache__/"
)

$filesToAdd = @($sourceFiles)

if ($IncludeRuntimeArtifacts) {
    $filesToAdd += $runtimeFiles
    $filesToAdd += $pycachePaths
}

foreach ($path in $filesToAdd) {
    if (Test-Path $path) {
        git add -- $path
    }
}

git status --short
