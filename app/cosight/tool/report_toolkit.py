import concurrent.futures
import datetime
import glob
import json
import os
import re
import threading
import traceback
from collections import Counter
from difflib import SequenceMatcher
from urllib.parse import urlparse

from app.common.logger_util import logger


MAX_CONTENT_LENGTH = int(os.environ.get("REPORT_MAX_CONTENT_LENGTH", "120000"))
MAX_FILE_PREVIEW_LENGTH = int(os.environ.get("REPORT_MAX_FILE_PREVIEW_LENGTH", "12000"))
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.bmp")
URL_PATTERN = re.compile(r"https?://[^\s<>\]\)\"'，。；、]+", re.IGNORECASE)


INTERNAL_COMPRESSION_PATTERNS = (
    re.compile(
        r"\n?\[Compressed before tool call: original \d+ chars, kept <= \d+ chars\.[^\]]*\]\s*",
        re.IGNORECASE,
    ),
    re.compile(r"\n?\[Content compressed: middle omitted to keep tool call complete\.\]\s*", re.IGNORECASE),
)


def strip_internal_compression_markers(content):
    cleaned = str(content or "")
    for pattern in INTERNAL_COMPRESSION_PATTERNS:
        cleaned = pattern.sub("\n", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


class ReportToolkit:
    """Markdown report generator following the HTML report workflow."""

    def __init__(self, workspace_path=None, tool_llm=None):
        self.workspace_path = workspace_path if workspace_path else os.environ.get("WORKSPACE_PATH") or os.getcwd()
        if tool_llm:
            self.llm_for_tool = tool_llm
        else:
            from llm import llm_for_tool
            self.llm_for_tool = llm_for_tool

    def get_workspace_path(self):
        return self.workspace_path or os.environ.get("WORKSPACE_PATH") or os.getcwd()

    def ask_llm(self, prompt):
        try:
            return self.llm_for_tool.chat_to_llm([{"role": "user", "content": prompt}])
        except Exception as e:
            logger.error(f"Failed to call LLM for report generation: {str(e)}", exc_info=True)
            return None

    def read_text_files_from_workspace(self):
        workspace_path = self.get_workspace_path()
        text_files = []

        for ext in ("*.txt", "*.md", "*.json", "*.csv"):
            for file_path in glob.glob(os.path.join(workspace_path, ext)):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, "r", encoding="gbk", errors="ignore") as f:
                            content = f.read()
                    except Exception as e:
                        logger.error(f"Failed to read file {file_path}: {str(e)}")
                        continue
                except Exception as e:
                    logger.error(f"Failed to read file {file_path}: {str(e)}")
                    continue

                content = strip_internal_compression_markers(content)
                text_files.append({
                    "path": file_path,
                    "filename": os.path.basename(file_path),
                    "content": content
                })

        text_files = self._dedupe_text_files(text_files)
        logger.info(f"Found {len(text_files)} unique text files in workspace: {workspace_path}")
        return text_files

    def read_image_files_from_workspace(self):
        workspace_path = self.get_workspace_path()
        image_files = []
        seen = set()
        for ext in IMAGE_EXTENSIONS:
            for file_path in glob.glob(os.path.join(workspace_path, ext)):
                filename = os.path.basename(file_path)
                if filename in seen:
                    continue
                seen.add(filename)
                image_files.append({
                    "path": file_path,
                    "filename": filename
                })
        logger.info(f"Found {len(image_files)} image files in workspace: {workspace_path}")
        return image_files

    def collect_web_references(self, text_files):
        references = []
        seen_urls = set()
        for file_info in text_files:
            content = file_info.get("content", "")
            markdown_links = {
                self._clean_url(match.group(2)): self._clean_reference_title(match.group(1))
                for match in re.finditer(r"\[([^\]\n]{2,200})\]\((https?://[^)\s]+)\)", content)
            }
            for match in URL_PATTERN.finditer(content):
                url = self._clean_url(match.group(0))
                if not url or url in seen_urls or self._is_non_reference_url(url):
                    continue
                seen_urls.add(url)
                title = markdown_links.get(url) or self._infer_reference_title(content, match.start(), match.end(), url)
                references.append({
                    "id": len(references) + 1,
                    "title": title,
                    "url": url,
                    "source_file": file_info.get("filename", "")
                })
        return references

    def _dedupe_text_files(self, text_files):
        unique_files = []
        seen_content = set()
        seen_filenames = set()
        for file_info in text_files:
            content = self._dedupe_repeated_blocks(file_info.get("content", ""))
            normalized_content = re.sub(r"\s+", " ", content).strip()
            content_key = normalized_content[:20000]
            filename = file_info.get("filename", "")
            if content_key and content_key in seen_content:
                logger.info(f"Skip duplicate source file content: {file_info.get('path')}")
                continue
            if filename in seen_filenames and content_key:
                logger.info(f"Skip duplicate source filename: {file_info.get('path')}")
                continue
            seen_content.add(content_key)
            seen_filenames.add(filename)
            file_info["content"] = content
            unique_files.append(file_info)
        return unique_files

    def _dedupe_repeated_blocks(self, content):
        if not content or len(content) < 2000:
            return content
        blocks = re.split(r"(?=^#{1,3}\s+)", content, flags=re.MULTILINE)
        if len(blocks) <= 2:
            return content
        seen = set()
        deduped = []
        for block in blocks:
            normalized = re.sub(r"\s+", " ", block).strip()
            if normalized and len(normalized) > 80:
                key = normalized[:4000]
                if key in seen:
                    continue
                seen.add(key)
            deduped.append(block)
        return "".join(deduped)

    def process_content_for_llm(self, content, _purpose="report"):
        if len(content) <= MAX_CONTENT_LENGTH:
            return content
        head_len = MAX_CONTENT_LENGTH * 3 // 4
        tail_len = MAX_CONTENT_LENGTH - head_len
        return (
            content[:head_len]
            + "\n\n[Content truncated due to length; preserving tail context below.]\n\n"
            + content[-tail_len:]
        )

    def generate_outline(self, text_files, user_query=""):
        all_content = ""
        for file in text_files:
            preview = file["content"][:MAX_FILE_PREVIEW_LENGTH]
            if len(file["content"]) > MAX_FILE_PREVIEW_LENGTH:
                preview += "..."
            all_content += f"File name: {file['filename']}\nContent preview:\n{preview}\n\n"

        processed_content = self.process_content_for_llm(all_content, "outline")
        is_chinese = bool(re.search(r"[\u4e00-\u9fff]", user_query)) if user_query else True

        if is_chinese:
            prompt = f"""请基于以下工作区文件内容生成一个结构清晰的 Markdown 报告大纲。

要求：
1. 包含主标题和副标题。
2. 包含 3-6 个主要章节。
3. 每个主要章节包含 2-4 个子章节。
4. content_from 必须列出该子章节应使用的来源文件名。

用户需求：
{user_query}

文件内容：
{processed_content}

请只返回 JSON，格式如下：
{{
  "title": "主标题",
  "subtitle": "副标题",
  "sections": [
    {{
      "title": "1 章节标题",
      "subsections": [
        {{"title": "1.1 子章节标题", "content_from": ["filename.md"]}}
      ]
    }}
  ]
}}"""
        else:
            prompt = f"""Generate a clear Markdown report outline based on the following workspace files.

Requirements:
1. Include a main title and subtitle.
2. Include 3-6 main sections.
3. Include 2-4 subsections under each main section.
4. content_from must list the source filenames for each subsection.
5. Write the outline in the same language as the user request.

User request:
{user_query}

File content:
{processed_content}

Return JSON only:
{{
  "title": "Main title",
  "subtitle": "Subtitle",
  "sections": [
    {{
      "title": "1 Section title",
      "subsections": [
        {{"title": "1.1 Subsection title", "content_from": ["filename.md"]}}
      ]
    }}
  ]
}}"""

        outline = self._extract_json(self.ask_llm(prompt))
        if not outline:
            return None

        for section in outline.get("sections", []):
            for subsection in section.get("subsections", []):
                content_from = subsection.get("content_from", [])
                if not isinstance(content_from, list):
                    subsection["content_from"] = [content_from]
        return outline

    def reorganize_content(self, text_files, outline, user_query="", references=None, image_files=None):
        filename_to_content = {file["filename"]: file["content"] for file in text_files}
        references = references or []
        image_files = image_files or []
        sections = []
        section_lock = threading.Lock()
        is_chinese = bool(re.search(r"[\u4e00-\u9fff]", user_query)) if user_query else True

        for section in outline.get("sections", []):
            sections.append({
                "title": section.get("title", ""),
                "subsections": [None] * len(section.get("subsections", []))
            })

        def process_subsection(section_idx, subsection_idx, subsection):
            source_files = subsection.get("content_from", [])
            relevant_content = ""
            for filename in source_files:
                if filename in filename_to_content:
                    relevant_content += f"\n\n--- {filename} ---\n{filename_to_content[filename]}"

            if not relevant_content:
                relevant_content = "\n\n".join(filename_to_content.values())

            relevant_content = self.process_content_for_llm(relevant_content, "reorganize")
            relevant_references = self._references_for_content(relevant_content, references)
            if not relevant_references:
                relevant_references = references[:30]
            reference_registry = self._format_reference_registry(relevant_references, is_chinese)
            image_registry = self._format_image_registry(image_files, relevant_content, is_chinese)
            if is_chinese:
                prompt = f"""请根据原始内容撰写报告子章节“{subsection.get('title', '')}”。

要求：
1. 直接输出 Markdown 正文，不要包裹代码块。
2. 只输出当前子章节的正文，不要输出任何 Markdown 标题（`#` 至 `######`），不要重复子章节标题。
3. 保留原始事实、数字、结论和关键细节，不要杜撰。
4. 内容应结构清晰、适合作为最终报告的一部分。
5. 必须使用与用户需求相同的语言撰写；如果用户需求是德语、法语等非中文语言，也要使用该语言。
6. 事实、数据和结论句末应使用下方“可用参考文献”中的编号引用，如 `[1]`、`[2][3]`；不要引用本地中间过程文件名。
7. 禁止生成目录、摘要、总结、结论、参考文献或来源列表；最终报告会统一生成这些结构。
8. 如果原始内容提到本地图片文件，且该图片与本段论述直接相关，请在首次讨论该图片的位置后插入 Markdown 图片：`![image](文件名)`，并添加一行简短图注。图片路径只使用文件名。不要把同一张图片重复用作不同图表或示意图。

用户需求：
{user_query}

可用参考文献：
{reference_registry}

可用本地图片：
{image_registry}

原始内容：
{relevant_content}"""
            else:
                prompt = f"""Write the report subsection "{subsection.get('title', '')}" from the original content.

Requirements:
1. Return Markdown body content directly, without wrapping it in a code block.
2. Return only the current subsection body. Do not output any Markdown headings (`#` through `######`) or repeat the subsection title.
3. Preserve original facts, numbers, conclusions, and key details. Do not invent information.
4. Make it clear, structured, and suitable for a final report.
5. Write in the same language as the user request. If the user request is in German, French, Spanish, or another language, use that language for the subsection.
6. Add numbered citations at the end of factual, numerical, or conclusion sentences using the available references below, such as `[1]` or `[2][3]`. Do not cite local intermediate filenames.
7. Do not add a table of contents, abstract, summary, conclusion, references, bibliography, or source list; the final report renderer owns those structures.
8. If the original content mentions a local image file and the image directly supports the discussion, insert it at the first relevant place using `![image](filename)` and add one concise caption line. Use only the filename as the relative path. Do not reuse the same image as multiple different charts or illustrations.

User request:
{user_query}

Available references:
{reference_registry}

Available local images:
{image_registry}

Original content:
{relevant_content}"""

            draft = self.ask_llm(prompt) or ""
            return section_idx, subsection_idx, {
                "title": subsection.get("title", ""),
                "content": self._sanitize_subsection_body(draft)
            }

        tasks = []
        for section_idx, section in enumerate(outline.get("sections", [])):
            for subsection_idx, subsection in enumerate(section.get("subsections", [])):
                tasks.append((section_idx, subsection_idx, subsection))

        worker_limit = max(1, int(os.environ.get("REPORT_MAX_WORKERS", "4")))
        max_workers = min(worker_limit, max(1, min(os.cpu_count() or 4, len(tasks) or 1)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_subsection, *task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                section_idx, subsection_idx, subsection_data = future.result()
                with section_lock:
                    sections[section_idx]["subsections"][subsection_idx] = subsection_data

        logger.info("Sequentially merging subsection drafts with an already-covered claims ledger")
        return self._merge_subsection_drafts_sequentially(sections, user_query)

    def _sanitize_subsection_body(self, content, subsection_title=""):
        """Keep subsection prose while removing nested report structures deterministically."""
        content = self._unwrap_markdown_code_block(str(content or "")).strip()
        if not content:
            return ""

        title_key = self._normalize_heading_key(subsection_title)
        cleaned_lines = []
        skip_forbidden_level = None
        skip_unheaded_tail = False
        in_code_fence = False

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_fence = not in_code_fence
                if not skip_unheaded_tail and skip_forbidden_level is None:
                    cleaned_lines.append(line.rstrip())
                continue
            if in_code_fence:
                if not skip_unheaded_tail and skip_forbidden_level is None:
                    cleaned_lines.append(line.rstrip())
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if heading_match:
                level = len(heading_match.group(1))
                heading_title = heading_match.group(2).strip()
                if skip_forbidden_level is not None and level > skip_forbidden_level:
                    continue
                skip_forbidden_level = None
                skip_unheaded_tail = False
                if self._is_forbidden_subsection_heading(heading_title):
                    skip_forbidden_level = level
                # All subsection headings are renderer-owned, so the heading
                # line itself is always removed even when its prose is kept.
                continue

            if skip_forbidden_level is not None or skip_unheaded_tail:
                continue
            if re.match(r"^\s*[-*+]\s+\[[^\]]+\]\(#[^)]+\)\s*$", line):
                continue

            plain_label = re.sub(r"^[\s>*_`~\-]+|[\s:*_`~\-]+$", "", stripped)
            if len(plain_label) <= 80 and self._is_forbidden_subsection_heading(plain_label):
                skip_unheaded_tail = True
                continue
            if title_key and self._normalize_heading_key(plain_label) == title_key:
                continue
            cleaned_lines.append(line.rstrip())

        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    @staticmethod
    def _is_forbidden_subsection_heading(title):
        normalized = re.sub(r"^[\s\d.()（）一二三四五六七八九十①-⑳、:-]+", "", str(title or ""))
        normalized = re.sub(r"[\s:：.。]+$", "", normalized).strip().lower()
        forbidden_prefixes = (
            "目录", "目錄", "摘要", "总结", "總結", "小结", "小結", "结论", "結論", "参考",
            "table of contents", "contents", "toc", "abstract", "summary", "conclusion",
            "references", "reference", "bibliography", "sources", "source list",
            "table des matières", "résumé", "conclusion", "références", "bibliographie", "sources",
            "inhaltsverzeichnis", "zusammenfassung", "fazit", "schluss", "quellen", "literaturverzeichnis",
            "índice", "resumen", "conclusión", "referencias", "bibliografía",
            "目次", "まとめ", "結論", "参考文献",
        )
        return any(
            normalized == prefix
            or normalized.startswith(prefix + " ")
            or normalized.startswith(prefix + ":")
            or normalized.startswith(prefix + "：")
            for prefix in forbidden_prefixes
        )

    def _merge_subsection_drafts_sequentially(self, sections, user_query=""):
        """Merge parallel drafts in report order with a compact global claims ledger."""
        covered_claims = []
        is_chinese = bool(re.search(r"[\u4e00-\u9fff]", user_query)) if user_query else True

        for section in sections:
            for subsection in section.get("subsections", []):
                if not subsection:
                    continue
                title = subsection.get("title", "")
                draft = self._sanitize_subsection_body(subsection.get("content", ""), title)
                if not draft:
                    subsection["content"] = ""
                    continue

                ledger_text = self._format_covered_claims(covered_claims)
                if is_chinese:
                    prompt = f"""你是报告的顺序合并器。请在不增加篇幅的前提下合并当前子章节草稿。

必须遵守：
1. `already_covered_claims` 中的事实、数字、证据和结论已经在前文完整出现，本节不得再次展开、换言改写或重新罗列；确有衔接需要时最多一句简短指代。
2. 只保留当前草稿中尚未覆盖的新事实、新证据和本节独有分析，并保留它们原有的编号引用、图片和表格。
3. body 只能是当前子章节正文，禁止任何 Markdown 标题、目录、摘要、总结、结论、参考文献或来源列表。
4. 不得添加新事实、新引用或新图片；合并后的 body 不得比原草稿更长。
5. 保持原草稿的语言。new_claims 只列出本节新引入的核心事实或结论，最多 8 条，每条不超过 180 字。

请只返回 JSON：
{{"body": "合并后的正文", "new_claims": ["新主张1", "新主张2"]}}

用户需求：
{user_query}

already_covered_claims：
{ledger_text or "（无）"}

当前子章节：{title}

并行草稿：
{draft}"""
                else:
                    prompt = f"""You are the sequential merger for a report. Merge the current subsection draft without expanding it.

Mandatory rules:
1. Facts, numbers, evidence, and conclusions in `already_covered_claims` have already been fully stated. Do not restate, paraphrase, or relist them. If continuity requires it, use at most one short cross-reference sentence.
2. Keep only facts, evidence, and analysis unique to this subsection, preserving their existing numbered citations, images, and tables.
3. `body` must contain subsection prose only. Do not output Markdown headings, a table of contents, abstract, summary, conclusion, references, bibliography, or source list.
4. Add no new facts, citations, or images. The merged body must not be longer than the draft.
5. Keep the draft's language. `new_claims` must contain at most eight concise claims newly introduced here, each no longer than 180 characters.

Return JSON only:
{{"body": "merged subsection body", "new_claims": ["new claim 1", "new claim 2"]}}

User request:
{user_query}

already_covered_claims:
{ledger_text or "(none)"}

Current subsection: {title}

Parallel draft:
{draft}"""

                response = self.ask_llm(prompt)
                parsed = self._extract_json(response)
                candidate = ""
                new_claims = []
                if isinstance(parsed, dict):
                    candidate = self._sanitize_subsection_body(parsed.get("body", ""), title)
                    raw_claims = parsed.get("new_claims", [])
                    if isinstance(raw_claims, list):
                        new_claims = [str(claim).strip() for claim in raw_claims if str(claim).strip()]

                if not self._is_valid_merged_subsection(candidate, draft):
                    logger.warning(f"Sequential merge failed validation for subsection: {title}")
                    candidate = draft
                    new_claims = []

                subsection["content"] = candidate
                if not new_claims:
                    new_claims = self._extract_compact_claims(candidate)
                self._extend_covered_claims(covered_claims, new_claims)

        return sections

    @staticmethod
    def _is_valid_merged_subsection(candidate, draft):
        if not candidate or len(candidate) > len(draft):
            return False
        if re.search(r"(?m)^#{1,6}\s+", candidate):
            return False
        draft_citations = set(re.findall(r"\[(\d+)\]", draft))
        candidate_citations = set(re.findall(r"\[(\d+)\]", candidate))
        if not candidate_citations.issubset(draft_citations):
            return False
        draft_images = set(re.findall(r"!\[[^\]]*\]\(\s*([^)]+?)\s*\)", draft))
        candidate_images = set(re.findall(r"!\[[^\]]*\]\(\s*([^)]+?)\s*\)", candidate))
        return candidate_images.issubset(draft_images)

    def _extract_compact_claims(self, content, limit=8):
        claims = []
        content = re.sub(r"(?m)^#{1,6}\s+.+?$", "", str(content or ""))
        for part in self._split_sentences_preserving(content):
            claim = re.sub(r"^\s*(?:[-*+]|\d{1,3}[.)、])\s+", "", part.strip())
            if not claim or claim.startswith(("|", "![", "```")):
                continue
            normalized = self._normalize_repetition_unit(claim)
            effective_length = len(normalized) + len(re.findall(r"[\u3400-\u9fff]", normalized))
            if effective_length < 40:
                continue
            claim = re.sub(r"\s+", " ", claim)[:240].strip()
            if claim:
                claims.append(claim)
            if len(claims) >= limit:
                break
        return claims

    def _extend_covered_claims(self, covered_claims, new_claims, max_claims=48, max_chars=6000):
        for claim in new_claims:
            claim = re.sub(r"\s+", " ", str(claim or "")).strip()[:240]
            if not claim:
                continue
            normalized = self._normalize_repetition_unit(claim)
            if any(
                SequenceMatcher(
                    None,
                    normalized,
                    self._normalize_repetition_unit(existing),
                    autojunk=False,
                ).ratio() >= 0.92
                for existing in covered_claims
            ):
                continue
            if len(covered_claims) >= max_claims:
                break
            if sum(len(existing) for existing in covered_claims) + len(claim) > max_chars:
                break
            covered_claims.append(claim)

    @staticmethod
    def _format_covered_claims(covered_claims):
        return "\n".join(f"- {claim}" for claim in covered_claims)

    def save_markdown_report(self, markdown_content, report_name="report"):
        workspace_path = self.get_workspace_path()
        os.makedirs(workspace_path, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_report_name = re.sub(r'[\\/:*?"<>|]', "_", report_name or "report").strip(" ._")
        safe_report_name = safe_report_name or "report"
        filepath = os.path.join(workspace_path, f"{safe_report_name}_{timestamp}.md")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            logger.info(f"Markdown report saved to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save Markdown report: {str(e)}", exc_info=True)
            return None

    def generate_markdown_report(self, title=None, output_filename=None, user_query="", include_sources=True):
        """Create a Markdown report from workspace text files."""
        try:
            logger.info("=" * 50)
            logger.info("Start creating Markdown report")
            logger.info("=" * 50)

            logger.info("[Step 1/5] Reading workspace text files")
            text_files = self.read_text_files_from_workspace()
            image_files = self.read_image_files_from_workspace()
            if not text_files:
                return {
                    "status": "error",
                    "message": "No text files were found in the workspace. Please save source material as .txt, .md, .json, or .csv first."
                }
            references = self.collect_web_references(text_files)
            logger.info(f"Collected {len(references)} web references from workspace text files")
            logger.info("[Step 2/5] Generating report outline")
            outline = self.generate_outline(text_files, user_query)
            if outline is None:
                return {
                    "status": "error",
                    "message": "Failed to generate the report outline from the workspace files."
                }

            if title:
                outline["title"] = title

            logger.info("[Step 3/5] Reorganizing content according to outline")
            sections = self.reorganize_content(text_files, outline, user_query, references, image_files)
            if not sections:
                return {
                    "status": "error",
                    "message": "Failed to reorganize report content."
                }

            logger.info("[Step 4/6] Rendering Markdown report")
            markdown_content = self._generate_markdown_report(
                outline, sections, references, include_sources, user_query, image_files
            )

            logger.info("[Step 5/6] Polishing full Markdown report")
            markdown_content = self._finalize_markdown_report(
                markdown_content, user_query, references, image_files
            )
            logger.info("[Step 6/7] Applying report structure gate")
            markdown_content, structure_ok, structure_message = self._enforce_report_structure(
                markdown_content,
                outline,
                user_query=user_query,
                include_sources=include_sources,
                references=references,
            )
            if not structure_ok:
                return {
                    "status": "error",
                    "message": f"Report structure gate rejected the report: {structure_message}",
                }

            logger.info("[Step 7/7] Saving Markdown report")
            filename_prefix = output_filename or outline.get("title", "report").replace(" ", "_")
            report_path = self.save_markdown_report(markdown_content, filename_prefix)
            if not report_path:
                return {
                    "status": "error",
                    "message": "Failed to save the Markdown report."
                }

            return {
                "status": "success",
                "message": f"Markdown report has been generated and saved to: {report_path}",
                "report_path": report_path
            }
        except Exception as e:
            logger.error(f"Markdown report generation failed: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": f"Failed to create Markdown report: {str(e)}"
            }

    def _generate_markdown_report(
            self, outline, sections, references=None, include_sources=True, user_query="", image_files=None):
        title = self._clean_markdown_inline(outline.get("title") or "Report")
        subtitle = self._clean_markdown_inline(outline.get("subtitle") or "")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        references = references or []
        image_files = image_files or []
        report_language = self._detect_report_language(user_query or title or subtitle)
        is_chinese = report_language == "zh"
        labels = self._report_labels(report_language)
        reference_heading = labels["references"]

        lines = [f"# {title}", ""]
        if subtitle:
            lines.extend([subtitle, ""])
        lines.extend([f"> {labels['generated_at']}: {now}", ""])

        toc = self._generate_toc(
            sections,
            include_references=include_sources and bool(references),
            reference_heading=reference_heading
        )
        if toc:
            lines.extend([f"## {labels['toc']}", "", *toc, ""])

        for section_index, section in enumerate(sections, start=1):
            section_title = self._strip_leading_number(section.get("title") or f"Section {section_index}")
            lines.extend([f"## {self._clean_markdown_inline(section_title)}", ""])

            for subsection_index, subsection in enumerate(section.get("subsections", []), start=1):
                if subsection is None:
                    continue
                subsection_title = self._strip_leading_number(
                    subsection.get("title") or f"{section_index}.{subsection_index}"
                )
                content = self._sanitize_subsection_body(
                    subsection.get("content") or "", subsection_title
                )
                content = self._normalize_markdown_body(content)
                lines.extend([f"### {self._clean_markdown_inline(subsection_title)}", ""])
                lines.extend([content or "No content was generated for this subsection.", ""])

        report_body = "\n".join(lines)
        report_body = self._normalize_image_embeddings(report_body, image_files, is_chinese)
        lines = report_body.split("\n")

        references = self._renumber_references(references)
        if include_sources and references:
            lines.extend([f"## {reference_heading}", ""])
            for ref in references:
                title = self._reference_display_title(ref)
                url = ref.get("url", "")
                lines.extend([f"- [{ref['id']}] {title} - {url}", ""])
            lines.append("")
            for ref in references:
                lines.extend([f"[{ref['id']}]: {ref.get('url', '')}", ""])
        return "\n".join(lines)

    def _finalize_markdown_report(self, markdown_content, user_query="", references=None, image_files=None):
        """Run a final whole-document cleanup to reduce repeated blocks and broken structure."""
        references = references or []
        image_files = image_files or []
        body, reference_section = self._split_reference_section(markdown_content)
        body = self._repair_structural_repetition(body)

        enabled = os.environ.get("REPORT_ENABLE_FINAL_POLISH", "true").lower() not in ("0", "false", "no")
        if not enabled:
            return body + reference_section

        max_chars = max(4000, int(os.environ.get("REPORT_POLISH_MAX_CHARS", "60000")))
        chunks = self._split_markdown_for_polish(body, max_chars)
        if len(chunks) > 1:
            logger.info(
                f"Polishing long report in {len(chunks)} chunks: {len(body)} chars, "
                f"chunk limit {max_chars}"
            )

        image_names = ", ".join(img.get("filename", "") for img in image_files if img.get("filename"))
        polished_chunks = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_context = ""
            if len(chunks) > 1:
                chunk_context = (
                    f"This is chunk {chunk_index} of {len(chunks)} from one report. "
                    "Polish only this chunk and do not add a reference section.\n\n"
                )
            prompt = f"""You are polishing a final Markdown report before saving it.

Goals:
1. Repair broken Markdown hierarchy, duplicated sections, repeated paragraphs, and incoherent transitions.
2. Preserve the same language as the report and the user request.
3. Preserve all factual claims, citations such as `[1]`, local image embeds such as `![image](file.png)`, tables, and URLs already present.
4. Do not add new facts, new sources, generic disclaimers, or a new reference section.
5. Remove redundancy without expanding the text. The polished result must not be longer than the supplied Markdown body.
6. Return the complete polished Markdown body only, without a code block.

User request:
{user_query}

Local images that may already be embedded:
{image_names or "None"}

{chunk_context}Markdown body to polish:
{chunk}"""

            polished = self.ask_llm(prompt)
            polished = self._unwrap_markdown_code_block(str(polished or "")).strip()
            require_heading = chunk.lstrip().startswith("#")
            if self._is_valid_polished_body(polished, chunk, require_heading=require_heading):
                polished_chunks.append(polished)
            else:
                logger.warning(
                    f"Discarded final report polish for chunk {chunk_index}/{len(chunks)} "
                    "because the result failed validation"
                )
                polished_chunks.append(chunk)

        polished_body = "\n\n".join(chunk.strip() for chunk in polished_chunks if chunk.strip())
        if self._is_valid_polished_body(polished_body, body):
            return self._repair_structural_repetition(polished_body) + reference_section

        logger.warning("Discarded combined final report polish because the result failed validation")
        return body + reference_section

    def _enforce_report_structure(
        self,
        markdown_content,
        outline,
        user_query="",
        include_sources=True,
        references=None,
    ):
        """Reject structurally inflated reports unless targeted section rewrites repair them."""
        references = references or []
        issues = self._report_structure_issues(
            markdown_content, outline, user_query, include_sources, references
        )
        if not issues["has_issues"]:
            return markdown_content, True, "ok"

        logger.warning(f"Report structure gate triggered: {issues['message']}")
        rewritten = self._rewrite_abnormal_report_sections(
            markdown_content,
            outline,
            issues["problem_sections"],
            user_query,
        )
        remaining = self._report_structure_issues(
            rewritten, outline, user_query, include_sources, references
        )
        if remaining["has_issues"]:
            logger.error(f"Report structure gate still failing after targeted rewrite: {remaining['message']}")
            return rewritten, False, remaining["message"]
        return rewritten, True, "repaired"

    def _report_structure_issues(
        self,
        markdown_content,
        outline,
        user_query="",
        include_sources=True,
        references=None,
    ):
        expected = self._expected_report_structure(
            outline, user_query, include_sources, references or []
        )
        records = self._extract_heading_records(markdown_content)
        counts = Counter(record["key"] for record in records if record["key"])
        repeated_titles = {
            key: count
            for key, count in counts.items()
            if count >= 3 and count > expected["expected_counts"].get(key, 0)
        }
        unexpected = [record for record in records if record["key"] not in expected["allowed_keys"]]

        allowed_excess = max(3, expected["expected_count"] // 4)
        too_many_headings = len(records) > expected["expected_count"] + allowed_excess
        problem_sections = set()
        for record in unexpected:
            if record["parent_h2"] in expected["section_map"]:
                problem_sections.add(record["parent_h2"])
        for record in records:
            if record["key"] in repeated_titles and record["parent_h2"] in expected["section_map"]:
                problem_sections.add(record["parent_h2"])

        headings_per_section = Counter(
            record["parent_h2"]
            for record in records
            if record["level"] >= 3 and record["parent_h2"] in expected["section_map"]
        )
        for section_key, count in headings_per_section.items():
            expected_subsections = len(expected["section_map"][section_key]["subsection_keys"])
            if count > expected_subsections + 2:
                problem_sections.add(section_key)

        has_issues = bool(repeated_titles or too_many_headings)
        repeated_summary = ", ".join(
            f"{key} x{count}" for key, count in sorted(repeated_titles.items())[:8]
        )
        message = (
            f"headings={len(records)}, expected={expected['expected_count']}, "
            f"unexpected={len(unexpected)}, repeated={repeated_summary or 'none'}, "
            f"problem_sections={len(problem_sections)}"
        )
        return {
            "has_issues": has_issues,
            "message": message,
            "problem_sections": problem_sections,
        }

    def _expected_report_structure(self, outline, user_query, include_sources, references):
        title = self._clean_markdown_inline(outline.get("title") or "Report")
        report_language = self._detect_report_language(user_query or title)
        labels = self._report_labels(report_language)
        allowed_keys = {
            self._normalize_heading_key(title),
            self._normalize_heading_key(labels["toc"]),
        }
        expected_counts = Counter(allowed_keys)
        expected_count = 2  # H1 title and H2 table of contents
        section_map = {}

        for section_index, section in enumerate(outline.get("sections", []), start=1):
            section_title = self._strip_leading_number(
                section.get("title") or f"Section {section_index}"
            )
            section_key = self._normalize_heading_key(section_title)
            subsection_titles = []
            subsection_keys = []
            for subsection_index, subsection in enumerate(section.get("subsections", []), start=1):
                subsection_title = self._strip_leading_number(
                    subsection.get("title") or f"{section_index}.{subsection_index}"
                )
                subsection_titles.append(subsection_title)
                subsection_keys.append(self._normalize_heading_key(subsection_title))
            section_map[section_key] = {
                "title": section_title,
                "subsection_titles": subsection_titles,
                "subsection_keys": subsection_keys,
            }
            allowed_keys.add(section_key)
            allowed_keys.update(subsection_keys)
            expected_counts[section_key] += 1
            expected_counts.update(subsection_keys)
            expected_count += 1 + len(subsection_titles)

        if include_sources and references:
            reference_key = self._normalize_heading_key(labels["references"])
            allowed_keys.add(reference_key)
            expected_counts[reference_key] += 1
            expected_count += 1
        return {
            "allowed_keys": allowed_keys,
            "expected_counts": expected_counts,
            "expected_count": expected_count,
            "section_map": section_map,
        }

    def _extract_heading_records(self, markdown_content):
        records = []
        parent_h2 = ""
        for line_number, line in enumerate(str(markdown_content or "").splitlines(), start=1):
            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if not match:
                continue
            level = len(match.group(1))
            title = match.group(2).strip()
            key = self._normalize_heading_key(title)
            if level == 2:
                parent_h2 = key
            records.append({
                "line": line_number,
                "level": level,
                "title": title,
                "key": key,
                "parent_h2": parent_h2,
            })
        return records

    def _rewrite_abnormal_report_sections(
        self, markdown_content, outline, problem_sections, user_query=""
    ):
        expected = self._expected_report_structure(outline, user_query, False, [])
        matches = list(re.finditer(r"(?m)^##(?!#)\s+.+$", str(markdown_content or "")))
        if not matches:
            return markdown_content

        prefix = markdown_content[:matches[0].start()].rstrip()
        blocks = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown_content)
            blocks.append(markdown_content[match.start():end].strip())

        covered_claims = []
        rewritten_blocks = []
        is_chinese = bool(re.search(r"[\u4e00-\u9fff]", user_query)) if user_query else True
        for block in blocks:
            heading_match = re.match(r"^##\s+(.+?)\s*$", block.splitlines()[0])
            section_key = self._normalize_heading_key(heading_match.group(1)) if heading_match else ""
            section_spec = expected["section_map"].get(section_key)
            candidate = block

            if section_spec and section_key in problem_sections:
                allowed_headings = "\n".join(
                    [f"## {section_spec['title']}"]
                    + [f"### {title}" for title in section_spec["subsection_titles"]]
                )
                ledger_text = self._format_covered_claims(covered_claims)
                if is_chinese:
                    prompt = f"""请定向重写下面这个结构异常的报告章节。

要求：
1. 必须完整保留且只使用“允许的标题”；一级章节使用 `##`，子章节使用 `###`，禁止生成其他标题或 `####` 至 `######` 嵌套标题。
2. 删除重复标题、循环段落、目录、摘要、额外总结/结论和章节内参考文献。
3. `already_covered_claims` 已在前文完整出现，不得在本章节再次展开或换言复述。
4. 保留本章节独有的事实、数据、编号引用、图片和表格，不得添加新事实、新引用或新图片。
5. 重写结果不得比原章节更长。
6. 直接返回完整章节 Markdown，不要代码块。

允许的标题：
{allowed_headings}

already_covered_claims：
{ledger_text or "（无）"}

原章节：
{block}"""
                else:
                    prompt = f"""Rewrite the structurally abnormal report section below.

Requirements:
1. Preserve and use only the allowed headings. Use `##` for the section and `###` for its subsections. Do not create any other headings or nested `####` through `######` headings.
2. Remove repeated headings, looping prose, a table of contents, abstract, extra summary/conclusion, and section-level references.
3. Claims in `already_covered_claims` were fully covered earlier. Do not expand or paraphrase them again.
4. Preserve facts, data, numbered citations, images, and tables unique to this section. Add no new facts, citations, or images.
5. The rewritten result must not be longer than the original section.
6. Return the complete section Markdown directly, without a code block.

Allowed headings:
{allowed_headings}

already_covered_claims:
{ledger_text or "(none)"}

Original section:
{block}"""
                response = self.ask_llm(prompt)
                rewritten = self._unwrap_markdown_code_block(str(response or "")).strip()
                if self._is_valid_rewritten_section(rewritten, block, section_spec):
                    candidate = rewritten
                    logger.info(f"Targeted structure rewrite accepted for section: {section_spec['title']}")
                else:
                    logger.warning(f"Targeted structure rewrite failed validation: {section_spec['title']}")

            rewritten_blocks.append(candidate)
            if section_spec:
                self._extend_covered_claims(
                    covered_claims, self._extract_compact_claims(candidate, limit=10)
                )

        return "\n\n".join([prefix, *rewritten_blocks]).strip() + "\n"

    def _is_valid_rewritten_section(self, candidate, original, section_spec):
        if not self._is_valid_polished_body(candidate, original, require_heading=True):
            return False
        records = self._extract_heading_records(candidate)
        if not records or records[0]["level"] != 2:
            return False
        expected_section_key = self._normalize_heading_key(section_spec["title"])
        if records[0]["key"] != expected_section_key:
            return False
        if sum(1 for record in records if record["level"] == 2) != 1:
            return False
        if any(record["level"] not in (2, 3) for record in records):
            return False
        actual_subsections = [record["key"] for record in records if record["level"] == 3]
        if Counter(actual_subsections) != Counter(section_spec["subsection_keys"]):
            return False
        return True

    def _normalize_heading_key(self, title):
        cleaned = self._clean_markdown_inline(str(title or ""))
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"[*_`~]", "", cleaned)
        cleaned = re.sub(r"^\s*[①-⑳]\s*", "", cleaned)
        cleaned = self._strip_leading_number(cleaned)
        cleaned = re.sub(r"[\s:：.。]+$", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
        return cleaned

    def _split_markdown_for_polish(self, markdown_content, max_chars):
        """Split a long report at Markdown boundaries without changing the global context limit."""
        content = str(markdown_content or "").strip()
        if not content or len(content) <= max_chars:
            return [content] if content else []

        protected_parts = re.split(r"(```[\s\S]*?```)", content)
        blocks = []
        for part in protected_parts:
            if not part:
                continue
            if part.startswith("```"):
                blocks.append(part.strip())
                continue
            blocks.extend(
                block.strip()
                for block in re.split(r"\n{2,}|(?=^#{1,6}\s+)", part, flags=re.MULTILINE)
                if block.strip()
            )

        units = []
        for block in blocks:
            if len(block) <= max_chars:
                units.append(block)
                continue
            sentences = self._split_sentences_preserving(block)
            for sentence in sentences:
                sentence = sentence.strip()
                while len(sentence) > max_chars:
                    split_at = sentence.rfind(" ", int(max_chars * 0.75), max_chars)
                    if split_at < 0:
                        split_at = sentence.rfind("\n", int(max_chars * 0.75), max_chars)
                    if split_at < 0:
                        split_at = max_chars
                    units.append(sentence[:split_at].strip())
                    sentence = sentence[split_at:].strip()
                if sentence:
                    units.append(sentence)

        chunks = []
        current = []
        current_length = 0
        for unit in units:
            added_length = len(unit) + (2 if current else 0)
            if current and current_length + added_length > max_chars:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_length = 0
            current.append(unit)
            current_length += len(unit) + (2 if current_length else 0)
        if current:
            chunks.append("\n\n".join(current).strip())
        return chunks

    def _split_reference_section(self, markdown_content):
        lines = str(markdown_content or "").splitlines()
        reference_tokens = (
            "reference", "references", "quellen", "referencias", "r茅f", "r.f",
            "source", "sources", "鍙傝", "参考",
        )
        split_index = None
        for idx, line in enumerate(lines):
            if not line.startswith("## "):
                continue
            heading = line[3:].strip().lower()
            if any(token in heading for token in reference_tokens):
                split_index = idx
        if split_index is None:
            return str(markdown_content or "").rstrip() + "\n", ""
        body = "\n".join(lines[:split_index]).rstrip() + "\n\n"
        refs = "\n".join(lines[split_index:]).rstrip() + "\n"
        return body, refs

    def _repair_structural_repetition(self, markdown_content):
        content = self._unwrap_markdown_code_block(str(markdown_content or "")).strip()
        if not content:
            return content

        lines = content.splitlines()
        cleaned_lines = []
        previous_heading = None
        in_code_fence = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code_fence = not in_code_fence
                previous_heading = None
                cleaned_lines.append(line.rstrip())
                continue
            if in_code_fence:
                cleaned_lines.append(line.rstrip())
                continue
            heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if heading_match:
                heading_key = (
                    len(heading_match.group(1)),
                    re.sub(r"\s+", " ", heading_match.group(2).strip().lower()),
                )
                if heading_key == previous_heading:
                    continue
                previous_heading = heading_key
            elif line.strip():
                previous_heading = None
            cleaned_lines.append(line.rstrip())

        content = "\n".join(cleaned_lines)
        protected_blocks = {}

        def protect_code_block(match):
            placeholder = f"__COSIGHT_PROTECTED_BLOCK_{len(protected_blocks)}__"
            protected_blocks[placeholder] = match.group(0)
            return placeholder

        content = re.sub(r"```[\s\S]*?```", protect_code_block, content)
        blocks = re.split(r"\n{2,}", content)
        result = []
        paragraph_exact = set()
        paragraph_near = []
        sentence_exact = set()
        sentence_near = []
        list_exact = set()
        list_near = []
        for block in blocks:
            stripped = block.strip()
            if not stripped:
                continue
            if stripped in protected_blocks:
                result.append(stripped)
                continue
            if self._is_protected_markdown_block(stripped):
                result.append(stripped)
                continue

            filtered_lines = []
            for line in stripped.splitlines():
                list_match = re.match(
                    r"^(\s*(?:[-*+]|\d{1,4}[.)、]|[（(]\d+[）)])\s+)(.+)$",
                    line,
                )
                if list_match and self._is_repeated_text_unit(
                    list_match.group(2), list_exact, list_near, near_threshold=0.96
                ):
                    continue
                filtered_lines.append(line)
            stripped = "\n".join(filtered_lines).strip()
            if not stripped:
                continue

            sentence_parts = self._split_sentences_preserving(stripped)
            kept_sentences = []
            for sentence in sentence_parts:
                if self._is_repeated_text_unit(
                    sentence, sentence_exact, sentence_near, near_threshold=0.96
                ):
                    continue
                kept_sentences.append(sentence)
            stripped = "".join(kept_sentences).strip()
            if not stripped:
                continue

            if self._is_repeated_text_unit(
                stripped,
                paragraph_exact,
                paragraph_near,
                near_threshold=0.97,
                exact_min_chars=180,
                near_min_chars=220,
            ):
                continue
            result.append(stripped)

        repaired = "\n\n".join(result).strip()
        for placeholder, code_block in protected_blocks.items():
            repaired = repaired.replace(placeholder, code_block)
        return repaired + "\n"

    @staticmethod
    def _split_sentences_preserving(text):
        return [
            part
            for part in re.split(r"(?<=[。！？；])|(?<=[.!?;])(?=\s+|$)", str(text or ""))
            if part and part.strip()
        ]

    @staticmethod
    def _is_protected_markdown_block(block):
        lines = block.splitlines()
        return (
            block.startswith(("#", "|", "![", "["))
            or any(line.lstrip().startswith("|") for line in lines)
            or any(line.lstrip().startswith("__COSIGHT_PROTECTED_BLOCK_") for line in lines)
        )

    @staticmethod
    def _normalize_repetition_unit(text):
        normalized = re.sub(
            r"^\s*(?:[-*+]|\d{1,4}[.)、]|[（(]\d+[）)])\s+",
            "",
            str(text or ""),
        )
        normalized = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", normalized)
        normalized = re.sub(r"\[(\d+)\]", "[citation]", normalized)
        normalized = re.sub(r"[*_`~>#]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip().lower()
        return normalized

    @staticmethod
    def _repetition_guard_signature(text):
        raw = str(text or "").lower()
        raw_without_citations = re.sub(r"\[\d+\]", "", raw)
        urls = tuple(match.rstrip(".,;:!?，。；：！？") for match in URL_PATTERN.findall(raw))
        numbers = tuple(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", raw_without_citations))
        negations = tuple(
            re.findall(
                r"\b(?:not|no|never|without|except)\b|没有|并非|禁止|不|未|无|非|仅",
                raw_without_citations,
            )
        )
        return urls, numbers, negations

    def _is_repeated_text_unit(
        self,
        text,
        exact_seen,
        near_seen,
        near_threshold,
        exact_min_chars=48,
        near_min_chars=100,
    ):
        normalized = self._normalize_repetition_unit(text)
        # A CJK character usually carries more information than one Latin
        # character, so weight it twice when applying length safety gates.
        effective_length = len(normalized) + len(re.findall(r"[\u3400-\u9fff]", normalized))
        if effective_length < exact_min_chars:
            return False
        signature = self._repetition_guard_signature(text)
        exact_key = (normalized, signature)
        if exact_key in exact_seen:
            return True
        exact_seen.add(exact_key)

        if effective_length < near_min_chars:
            return False
        for previous, previous_signature in reversed(near_seen[-200:]):
            if signature != previous_signature:
                continue
            length_ratio = min(len(normalized), len(previous)) / max(len(normalized), len(previous))
            if length_ratio < near_threshold:
                continue
            matcher = SequenceMatcher(None, normalized, previous, autojunk=False)
            if matcher.quick_ratio() >= near_threshold and matcher.ratio() >= near_threshold:
                return True
        near_seen.append((normalized, signature))
        return False

    def _is_valid_polished_body(self, polished, original, require_heading=True):
        if not polished or (require_heading and not polished.lstrip().startswith("#")):
            return False
        if len(original) > 2000 and len(polished) < len(original) * 0.35:
            return False
        if len(polished) > len(original):
            return False
        original_images = set(re.findall(r"!\[[^\]]*\]\(\s*([^)]+?)\s*\)", original))
        polished_images = set(re.findall(r"!\[[^\]]*\]\(\s*([^)]+?)\s*\)", polished))
        if not original_images.issubset(polished_images):
            return False
        original_citations = set(re.findall(r"\[(\d+)\]", original))
        polished_citations = set(re.findall(r"\[(\d+)\]", polished))
        if original_citations and len(polished_citations) < max(1, int(len(original_citations) * 0.6)):
            return False
        return True

    def _generate_toc(self, sections, include_references=False, reference_heading="References"):
        toc = []
        for section in sections:
            if not section:
                continue
            title = self._strip_leading_number(section.get("title") or "")
            if not title:
                continue
            toc.append(f"- [{self._clean_markdown_inline(title)}](#{self._slugify(title)})")
            for subsection in section.get("subsections", []):
                if not subsection:
                    continue
                subtitle = self._strip_leading_number(subsection.get("title") or "")
                if subtitle:
                    toc.append(f"  - [{self._clean_markdown_inline(subtitle)}](#{self._slugify(subtitle)})")
        if include_references:
            toc.append(f"- [{reference_heading}](#{self._slugify(reference_heading)})")
        return toc

    def _normalize_markdown_body(self, content):
        content = str(content or "").strip()
        content = self._unwrap_markdown_code_block(content)
        content = re.sub(
            r"^(#{1,3})(\s+)",
            lambda m: "#" * (len(m.group(1)) + 3) + m.group(2),
            content,
            flags=re.MULTILINE
        )
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()

    def _unwrap_markdown_code_block(self, content):
        match = re.fullmatch(r"```(?:markdown|md)?\s*([\s\S]*?)\s*```", content.strip(), re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return content

    def _extract_json(self, text):
        if not text:
            return None
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        json_str = json_match.group(1) if json_match else text.strip()
        if not json_str.startswith("{"):
            object_match = re.search(r"\{[\s\S]*\}", json_str)
            json_str = object_match.group(0) if object_match else json_str
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse outline JSON: {str(e)}; raw response: {text}")
            return None

    def _strip_leading_number(self, text):
        return re.sub(r"^\s*\d+(?:\.\d+)*\s*[\.\-、)]?\s*", "", str(text or "")).strip()

    def _clean_markdown_inline(self, text):
        text = self._unwrap_markdown_code_block(str(text or ""))
        text = re.sub(r"[\r\n]+", " ", text)
        return text.strip()

    def _slugify(self, text):
        text = self._clean_markdown_inline(text).lower()
        text = re.sub(r"[^\w\u4e00-\u9fff\s-]", "", text)
        text = re.sub(r"\s+", "-", text).strip("-")
        return text or "section"

    def _detect_report_language(self, text):
        text = str(text or "")
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        lowered = text.lower()
        if re.search(r"[äöüß]", lowered) or re.search(r"\b(der|die|das|und|mit|für|bewerte|bericht|deutsch)\b", lowered):
            return "de"
        if re.search(r"[éèêëàâçîïôùûüÿœ]", lowered) or re.search(r"\b(le|la|les|des|avec|pour|rapport|évaluer|français)\b", lowered):
            return "fr"
        if re.search(r"[áéíóúñ¿¡]", lowered) or re.search(r"\b(el|la|los|las|con|para|informe|evaluar|español)\b", lowered):
            return "es"
        return "en"

    def _report_labels(self, language):
        labels = {
            "zh": {"generated_at": "生成时间", "toc": "目录", "references": "参考文献"},
            "de": {"generated_at": "Erstellt am", "toc": "Inhaltsverzeichnis", "references": "Quellen"},
            "fr": {"generated_at": "Généré le", "toc": "Table des matières", "references": "Références"},
            "es": {"generated_at": "Generado el", "toc": "Índice", "references": "Referencias"},
            "en": {"generated_at": "Generated at", "toc": "Table of Contents", "references": "Reference"},
        }
        return labels.get(language, labels["en"])

    def _clean_url(self, url):
        return str(url or "").strip().rstrip(".,;:!?，。；：！？)]）}\"'")

    def _is_non_reference_url(self, url):
        parsed = urlparse(url)
        if not parsed.netloc:
            return True
        lower = url.lower()
        return lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".css", ".js"))

    def _infer_reference_title(self, content, start, end, url):
        line_start = content.rfind("\n", 0, start) + 1
        line_end = content.find("\n", end)
        if line_end == -1:
            line_end = len(content)
        line = content[line_start:line_end]
        candidate = line.replace(url, " ")
        source_label_match = re.search(
            r"(?:来源|出处|参考|Source|Sources|Reference|Link|URL)\s*[:：-]\s*([^\n]{2,200})",
            candidate,
            flags=re.IGNORECASE
        )
        if source_label_match:
            labeled_candidate = self._clean_reference_title(source_label_match.group(1))
            if labeled_candidate:
                return labeled_candidate
        candidate = re.sub(r"^\s*[-*+\d.\])、]+\s*", "", candidate).strip()
        candidate = re.sub(r"(?i)\b(url|link|source|sources)\b\s*[:：-]?\s*", "", candidate).strip()
        candidate = re.sub(r"(来源|链接|网址)\s*[:：-]?\s*", "", candidate).strip()
        candidate = self._clean_reference_title(candidate)
        if candidate:
            return candidate

        window = content[max(0, start - 500):end]
        patterns = [
            r"['\"]title['\"]\s*:\s*['\"]([^'\"]{2,200})['\"]",
            r"标题\s*[:：]\s*([^\n]{2,200})",
            r"Title\s*[:：]\s*([^\n]{2,200})",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, window, flags=re.IGNORECASE)
            if matches:
                candidate = self._clean_reference_title(matches[-1])
                if candidate:
                    return candidate

        parsed = urlparse(url)
        return parsed.netloc or url

    def _clean_reference_title(self, title):
        title = self._unwrap_markdown_code_block(str(title or ""))
        title = re.sub(r"https?://\S+", "", title)
        title = re.sub(r"[*_`#>]+", " ", title)
        title = re.sub(r"[\r\n\t]+", " ", title)
        title = re.sub(r"\s+", " ", title).strip(" -:：|[]()（）")
        if (
                not title
                or title.lower() in {"url", "link", "source", "sources"}
                or re.fullmatch(r"\d+", title)
                or not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", title)
        ):
            return ""
        return title[:200]

    def _reference_display_title(self, ref):
        title = self._clean_reference_title(ref.get("title") or "")
        if title:
            return self._clean_markdown_inline(title)
        parsed = urlparse(ref.get("url", ""))
        return parsed.netloc or ref.get("url", "")

    def _renumber_references(self, references):
        deduped = []
        seen = set()
        for ref in references or []:
            url = self._clean_url(ref.get("url", ""))
            if not url or url in seen:
                continue
            seen.add(url)
            new_ref = dict(ref)
            new_ref["url"] = url
            new_ref["title"] = self._reference_display_title(new_ref)
            new_ref["id"] = len(deduped) + 1
            deduped.append(new_ref)
        return deduped

    def _references_for_content(self, content, references):
        urls = set(self._clean_url(match.group(0)) for match in URL_PATTERN.finditer(content or ""))
        return [ref for ref in references if ref.get("url") in urls]

    def _format_reference_registry(self, references, is_chinese=False):
        if not references:
            return "无可用网页参考文献。" if is_chinese else "No web references available."
        lines = []
        for ref in references:
            title = self._clean_markdown_inline(ref.get("title") or ref.get("url") or "")
            lines.append(f"[{ref['id']}] {title} - {ref.get('url', '')}")
        return "\n".join(lines)

    def _format_image_registry(self, image_files, content="", is_chinese=False):
        if not image_files:
            return "无可用本地图片。" if is_chinese else "No local images available."
        mentioned = [img for img in image_files if img.get("filename") in (content or "")]
        if not mentioned:
            return "原始内容未提到本地图片；不要主动插入图片。" if is_chinese else "No local image is mentioned in the source content; do not insert images proactively."
        selected = mentioned
        return "\n".join(f"- {img.get('filename')}" for img in selected)

    def _normalize_image_embeddings(self, markdown_content, image_files, is_chinese=False):
        content = markdown_content
        for image in image_files:
            filename = image.get("filename", "")
            if not filename or filename not in content:
                continue
            image_pattern = re.compile(r"!\[[^\]]*\]\(\s*" + re.escape(filename) + r"\s*\)")
            content = self._dedupe_image_embedding(content, image_pattern)
            if image_pattern.search(content):
                continue
            lines = content.split("\n")
            inserted = False
            for idx, line in enumerate(lines):
                if filename in line and not line.lstrip().startswith("!["):
                    caption = f"图像 {os.path.splitext(filename)[0]}。" if is_chinese else f"Figure {os.path.splitext(filename)[0]}."
                    lines[idx + 1:idx + 1] = ["", f"![image]({filename})", caption, ""]
                    inserted = True
                    break
            if inserted:
                content = "\n".join(lines)
        return content

    def _dedupe_image_embedding(self, content, image_pattern):
        lines = content.split("\n")
        result = []
        seen = False
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if image_pattern.search(line):
                if not seen:
                    result.append(line)
                    seen = True
                    idx += 1
                    continue

                idx += 1
                while idx < len(lines):
                    next_line = lines[idx]
                    stripped = next_line.strip()
                    if not stripped:
                        idx += 1
                        continue
                    if (
                            stripped.startswith(("图", "*图", "Figure", "*Figure", "Caption", "*Caption"))
                            or "图注" in stripped
                            or "caption" in stripped.lower()
                    ):
                        idx += 1
                        continue
                    break
                continue

            result.append(line)
            idx += 1
        return "\n".join(result)


def main(params=None):
    """Entry point used by the agent framework."""
    try:
        toolkit = ReportToolkit()
        params = params if isinstance(params, dict) else {}
        result = toolkit.generate_markdown_report(
            title=params.get("title") or params.get("report_title"),
            output_filename=params.get("output_filename"),
            user_query=params.get("user_query", ""),
            include_sources=params.get("include_sources", True)
        )
        logger.info(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except Exception as e:
        logger.error(f"Failed to run Markdown report tool: {traceback.format_exc()}")
        return {
            "status": "error",
            "message": f"Execution error: {str(e)}"
        }


if __name__ == "__main__":
    main()
