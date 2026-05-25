## 2024-03-25 - Avoid optimizing stub data
**Learning:** The prompt constraints for the Bolt performance agent clearly state to 'avoid premature optimization of cold paths' and 'micro-optimizations with no measurable impact'. Optimizing stub or mock data paths violates this rule.
**Action:** Verify if the code being optimized is part of a critical or hot path before applying optimizations. Ignore stubbed out implementations unless specifically tasked otherwise.
