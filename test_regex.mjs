// 模拟 LLM 实际输出：\(...\) 里面换行（多行行内公式），表格之间
const content = `
| f | Ui | Uc | phi |
|---|---|---|---|
| 1k | | | -90 |

$$
\\dfrac{U_i}{I} = \\dfrac{U_i}{U_C / X_C}
$$

| f | Ui | Uc | phi |
|---|---|---|---|
| 1k | ____ | ____ | — |
| 5k | ____ | ____ | — |
| 10k | ____ | ____ | — |

结论：\\(
Q = \\dfrac{U_i}{U_C}
\\)
`;

const result = content
  .replace(/\\\[([\s\S]+?)\\\]/g, (_, e) => `\n$$\n${e.trim()}\n$$\n`)
  .replace(/\\\(([\s\S]+?)\\\)/g, (_, e) => `$${e.trim()}$`)
  .replace(/([^\n])\$\$((?:(?!\n\n)[\s\S])+?)\$\$([^\n])/g, (_, pre, e, post) =>
    `${pre}\n$$\n${e.trim()}\n$$\n${post}`
  );

console.log("=== RESULT ===");
console.log(result);
console.log("=== TABLE 2 intact? ===", result.includes('5k') ? 'YES' : 'NO - EATEN');
console.log("=== TABLE 3 intact? ===", result.includes('10k') ? 'YES' : 'NO - EATEN');
