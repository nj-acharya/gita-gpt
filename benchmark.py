import time
from rag_hybrid_ui_quantize import generate

# Your test queries here
QUERIES = [
    "Explain the main teaching of Bhagavad Gita in simple terms",
    "Please summarize: The Mahabharata war lasted 18 days and involved...",
    "What is the importance of karma yoga?"
]

def benchmark(use_quantization):
    results = []
    print(f"\n=== Benchmark (USE_QUANTIZATION={use_quantization}) ===")
    for q in QUERIES:
        start = time.perf_counter()
        out = generate(q, max_tokens=200, use_rag=True)
        elapsed = time.perf_counter() - start
        results.append({
            "query": q,
            "time_sec": round(elapsed, 3),
            "output_len": len(out),
            "preview": out[:80].replace("\n", " ") + "..."
        })
        print(f"[{elapsed:.3f}s] {q} -> {out[:60]}...")
    return results

if __name__ == "__main__":
    # First run: quantization ON
    from rag_hybrid_ui_quantize import USE_QUANTIZATION
    USE_QUANTIZATION = True
    res_quant = benchmark(True)

    # Second run: quantization OFF
    USE_QUANTIZATION = False
    res_noquant = benchmark(False)

    # You can compare res_quant vs res_noquant here
    print("\n=== Summary ===")
    for i, q in enumerate(QUERIES):
        print(f"\nQuery: {q}")
        print(f"  Quantized: {res_quant[i]['time_sec']}s, len={res_quant[i]['output_len']}")
        print(f"  No-Quant:  {res_noquant[i]['time_sec']}s, len={res_noquant[i]['output_len']}")
