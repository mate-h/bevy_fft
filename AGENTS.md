# Agent notes on writing style

Source: https://gist.github.com/mate-h/3989935e364a501b2f855d0a08765ea8

Refer to this guide for Markdown in-repo docs, inline documentation, code comments, and UI copy.

## Tone

- Write in plain, readable English: short paragraphs, full sentences, the way you’d explain the project to another developer in person.
- Prefer **concrete descriptions** over changelog-style bullet dumps. Lists are fine when they help scanability, but avoid turning every section into `**Label**: explanation` rows.
- **Do not use emoji** in project writing unless someone **explicitly requests** them for that change. That includes Markdown, **inline documentation** tied to APIs or modules, ordinary **code comments**, and UI copy. No decorative or tonal icons instead of plain words.

## Punctuation

- Use **periods and new sentences** instead of stringing thoughts together with **semicolons**. Multiple `;` in one line often reads like stacked AI clauses. Break them apart. When two ideas could each stand alone, prefer two sentences over `clause one; clause two`, including in README and other short reader-facing blurbs.
- Do not lean on **em dashes (—)** or **hyphen-style pauses** to splice extra ideas into a sentence. If it feels like an aside or a second thought, make it the **next sentence**.

## Avoid “AI default” patterns

- **Title-colon explanations.** Patterns like `**Subsystem name**: the pipeline runs…` or opening lines that string together several internal pieces read robotic. Rewrite as normal sentences, or fold the idea into a short paragraph.
- **Heavy parenthetical asides.** Cut down `(example here)` and `(see above)`. Prefer another sentence or a light comma, or drop the aside.
- **“You get” / “you can”** as the main verb for describing what the **library or product** offers. Prefer **includes**, **ships with**, or a direct subject (**It includes…**, **The library ships with…**).
- **“Helpers”** when naming the project. Prefer **library** (or **package**) for what the project is.

## Lists and enumerations

- Prefer **commas** with **or** or **and** when you list **alternatives or modes in ordinary prose**. Example: *spatial, frequency, or hybrid mode* instead of *spatial/frequency/hybrid*. This habit targets **wordy lists** in sentences, not notation elsewhere.
- **Math expressions** may still use slashes, such as in **fractions** or **ratios**. Keep normal mathematical conventions.
- **Paths**, **grid sizes** like **256x256** or **N×M**, and similar shorthands are unchanged by this rule.
- For lists in prose, use **and** when everything applies together and **or** when the reader chooses one. Do not substitute `/` for those words in that kind of list.

## Documentation structure

- In README-style Markdown and other reader-facing docs, avoid leaning on bold for every API name, type, or keyword. Prefer `inline code` when an identifier needs to stand out. Use bold sparingly for real emphasis, not as a way to decorate each important-looking word.
- For document titles and major headings, prefer a **short connecting phrase** over a **parenthetical qualifier**. Example: *Agent notes on writing style* reads more naturally than *Agent notes (writing style)*, which looks like a filed label. Same idea as *Guide to X*, *Notes on Y*, *Z for the project*.
- Section headings can be neutral (**What it includes**, **Try it**) instead of marketing-style lead-ins.
- For **public APIs**, weave identifiers into sentences instead of serial **`Name`**: behavior bullets, unless a tight reference table is really needed.
- Link to source paths or your **published API reference** when it exists. Do not promise a **hosting URL** for docs until the project actually publishes there.

## Use case before implementation

- In READMEs, overviews, and other entry points, **lead with why someone would care**: the problem, the outcome, or the workflow they unlock. Low-level structure (call order, storage layout, private types) belongs in **later sections** or in the code, not as the hook.
- Do not **open** by inventorying how the program is built (every layer, pipeline step, or internal name) when a short outcome-focused sentence would answer “what is this for?”.

## UI copy

Use the same habits as above, plus normal UI wording rules.

- **Short fragments.** Prefer a few words over a sentence when space is tight. Cut filler.
- **Buttons and actions** should read like something the user can do. Use verbs or clear verb phrases where it fits (*Save*, *Open settings*, *Export*).
- **Labels** stay simple and descriptive. Name the thing or value, not a tutorial (*Center*, *Width*, *Gain* rather than *The center frequency of your band* in a small label).
- **Hints and help lines** can be one short line. Same punctuation habits as doc prose.

## Describe the present only

**Inline documentation** (comments or doc blocks attached to symbols or modules), **ordinary code comments**, and **UI strings** should describe the **current** behavior and intent. They are not a dev diary.

- Do not record **how or why something changed over time**, or **chat, review, or prompt** context (*we used to*, *changed from X to Y*, *per feedback*, *assistant said*). **Git history** holds that.
- When behavior or copy changes, **replace** the comment or UI text with a fresh description of the latest truth. Do not stack *Updated: …* lines or leave stale stories next to the new code.

## Code and technical prose

- Match the **surrounding codebase**: naming, comment density, and local conventions for imports or file layout.
- Prefer **accurate, calm** technical wording over hype. State limits and scope plainly (**building block**, not “complete solution”) when it avoids confusion.
