# Project Management & Automation

This directory contains project management tools and GitHub automation configurations for Quotient-Probes.

## Files

### `roadmap.md`
Mermaid Gantt chart tracking v0.1.2 - v0.1.4 milestones. Visualizes the Phase 2 execution plan from the PRD.

### `project-config.json`
Configuration for GitHub Projects automation. Maps issue/PR labels to project board columns.

## GitHub Workflows

### `.github/workflows/publish.yml`
**Auto-publish to PyPI on version tags**

**Setup Required:**
1. Create PyPI API token at https://pypi.org/manage/account/token/
2. Add to GitHub Secrets as `PYPI_API_TOKEN`:
   ```
   Settings → Secrets and variables → Actions → New repository secret
   Name: PYPI_API_TOKEN
   Value: pypi-[your-token-here]
   ```

**Usage:**
```bash
# Create and push version tag to trigger publish
git tag v0.1.2
git push origin v0.1.2
```

### `.github/workflows/project-automation.yml`
**Auto-add issues/PRs to GitHub project board**

**Setup Required (Optional):**
1. Create GitHub Personal Access Token (PAT) with `project` scope
2. Add secrets to repository:
   ```
   PROJECT_PAT: ghp_[your-token]
   PROJECT_URL: https://github.com/users/[username]/projects/[number]
   ```

**How it works:**
- New issues/PRs are automatically added to project board
- Status is set based on labels (configured in `project-config.json`)

**Configuration:**
Edit `project-config.json` to customize label → status mappings:
```json
{
  "bug": "To Do",
  "in-progress": "In Progress",
  "ready-for-review": "Review"
}
```

## Templates

### `.github/ISSUE_TEMPLATE/rfc.md`
RFC (Request for Comments) template for proposing breaking changes in v0.2+.

**When to use:**
- Proposing breaking API changes
- Major architectural changes
- New features requiring community feedback

## Development Workflow

1. **Create Issue** → Auto-added to project board (if configured)
2. **Work on Feature** → Update issue status with labels
3. **Create PR** → Auto-added to project board
4. **Merge PR** → Close issue
5. **Create Tag** → Auto-publish to PyPI (if version tag)

## Notes

- Workflows are optional and require secrets to function
- The roadmap is a living document - update as priorities shift
- RFC template is for v0.2+ only (v0.1.x is pre-1.0 and can break without RFC)
