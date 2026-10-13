#!/usr/bin/env python3
"""
Layer_A/scripts/test_sglang_logprobs.py
Extracts top-10 candidate tokens and logprobs from SGLang on port 18000,
verifying candidate competition and sampling margin calculation.
"""
import requests
import json
import math

def test_sglang_topk():
    url = "http://127.0.0.1:18000/v1/chat/completions"
    payload = {
        "model": "gpt-oss-20b",
        "messages": [{"role": "user", "content": "Return FLAG or APPROVE for an offshore wire transfer of $50,000."}],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 10
    }

    print("Connecting to SGLang endpoint at", url)
    try:
        res = requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ Failed to reach SGLang server at {url}: {e}")
        return

    if res.status_code != 200:
        print("❌ Error:", res.status_code, res.text)
        return

    data = res.json()
    choice = data["choices"][0]
    logprobs_content = choice.get("logprobs", {}).get("content", [])
    if not logprobs_content:
        print("⚠️ No logprobs returned in response!")
        return

    top_candidates = logprobs_content[0].get("top_logprobs", [])
    print(f"\n✅ Extracted {len(top_candidates)} Candidate Tokens from SGLang:")
    print(f"{'Rank':<5} | {'Token':<22} | {'LogProb':<10} | {'Probability':<12}")
    print("-" * 55)

    probs = []
    tokens = []
    for rank, cand in enumerate(top_candidates, 1):
        tok = cand.get("token", "").replace("\n", "\\n")
        lp = cand.get("logprob", -99.0)
        prob = math.exp(lp)
        probs.append(prob)
        tokens.append(tok)
        print(f"{rank:<5} | {tok:<22} | {lp:<10.4f} | {prob:<12.4f}")

    if len(probs) >= 2:
        margin = probs[0] - probs[1]
        fragile = margin < 0.05
        badge = "🚨 FRAGILE (M < 0.05)" if fragile else "✅ STABLE (M >= 0.05)"
        print("\n" + "=" * 55)
        print(f"Top-2 Sampling Margin : {margin:.4f} -> {badge}")
        print("=" * 55)

if __name__ == "__main__":
    test_sglang_topk()
