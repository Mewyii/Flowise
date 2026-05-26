import { Document } from '@langchain/core/documents'
import { MarkdownTextSplitter, MarkdownTextSplitterParams } from '@langchain/textsplitters'

export type EnhancedMarkdownTextSplitterParams = Partial<MarkdownTextSplitterParams> & {
    mergeSmallChunksEnabled?: boolean
    splitByHeaders?: string
}

export class EnhancedMarkdownTextSplitter extends MarkdownTextSplitter implements MarkdownTextSplitterParams {
    mergeSmallChunksEnabled: boolean = true
    splitByHeaders = 'disabled'

    constructor(fields?: EnhancedMarkdownTextSplitterParams) {
        super(fields)
        this.mergeSmallChunksEnabled = fields?.mergeSmallChunksEnabled ?? true
        this.splitByHeaders = fields?.splitByHeaders ?? 'disabled'
    }

    async splitText(text: string): Promise<string[]> {
        if (this.splitByHeaders === 'disabled') {
            const chunks = await super.splitText(text)
            return this.mergeAdjacentChunks(chunks)
        } else {
            return this.splitTextByHeaders(text, this.splitByHeaders, this)
        }
    }

    async splitDocuments(documents: Document[]): Promise<Document[]> {
        if (this.splitByHeaders === 'disabled') {
            const results: Document[] = []

            for (const [index, doc] of documents.entries()) {
                const chunks = await super.splitText(doc.pageContent)
                const mergedChunks = this.mergeAdjacentChunks(chunks)

                results.push(...this.getChunksWithMetaData(mergedChunks, doc.metadata, index))
            }

            return results
        } else {
            const results: Document[] = []

            for (const [index, doc] of documents.entries()) {
                const chunks = await this.splitTextByHeaders(doc.pageContent, this.splitByHeaders, this)
                results.push(...this.getChunksWithMetaData(chunks, doc.metadata, index))
            }

            return results
        }
    }

    protected getChunksWithMetaData(chunks: string[], metadata: any, docIndex: number): Document[] {
        const results: Document[] = []

        const total = chunks.length
        let currentCharacter = 1

        for (const [index, chunk] of chunks.entries()) {
            const source = metadata.source && metadata.source !== 'blob' ? metadata.source : undefined

            results.push({
                pageContent: chunk,
                metadata: {
                    ...metadata,
                    chunk: {
                        id: `${source ?? docIndex}-${index + 1}-${total}`,
                        number: index + 1,
                        total,
                        chars: {
                            from: currentCharacter,
                            to: currentCharacter + chunk.length - 1
                        }
                    }
                }
            })
            currentCharacter += chunk.length
        }
        return results
    }

    protected mergeAdjacentChunks(chunks: string[]): string[] {
        if (!this.mergeSmallChunksEnabled || this.chunkSize <= 0 || chunks.length === 0) {
            return chunks
        }

        const merged: string[] = []
        let current = chunks[0]

        for (let i = 1; i < chunks.length; i++) {
            const next = chunks[i]
            const overlap = this.findOverlapLength(current, next)
            const combinedLength = current.length + next.length - overlap

            if (combinedLength <= this.chunkSize) {
                current = this.mergeChunks(current, next, overlap)
            } else {
                merged.push(current)
                current = next
            }
        }

        merged.push(current)
        return merged
    }

    protected mergeChunks(current: string, next: string, overlap = 0): string {
        if (overlap > 0) {
            return `${current}${next.slice(overlap)}`
        }

        return `${current}\n\n${next}`
    }

    protected findOverlapLength(current: string, next: string): number {
        const maxOverlap = Math.min(this.chunkOverlap ?? 0, current.length, next.length)

        for (let length = maxOverlap; length > 0; length--) {
            if (current.endsWith(next.slice(0, length))) {
                return length
            }
        }

        return 0
    }

    protected async splitTextByHeaders(text: string, headerLevel: string, fallbackSplitter: any): Promise<string[]> {
        const maxLevel = this.getHeaderLevel(headerLevel)
        if (maxLevel === 0) return await fallbackSplitter.splitText(text)

        const lines = text.split('\n')
        const sections: string[] = []
        let currentSection: string[] = []

        for (const line of lines) {
            const isHeader = line.startsWith('#') && line.match(/^#{1,6}\s/)
            const headerDepth = isHeader ? line.match(/^(#+)/)?.[1]?.length || 0 : 0

            if (isHeader && headerDepth <= maxLevel) {
                if (currentSection.length > 0) {
                    sections.push(currentSection.join('\n').trim())
                }
                currentSection = [line]
            } else {
                currentSection.push(line)
            }
        }

        if (currentSection.length > 0) {
            sections.push(currentSection.join('\n').trim())
        }

        return sections
    }

    protected getHeaderLevel(headerLevel: string): number {
        switch (headerLevel) {
            case 'h1':
                return 1
            case 'h2':
                return 2
            case 'h3':
                return 3
            case 'h4':
                return 4
            case 'h5':
                return 5
            case 'h6':
                return 6
            default:
                return 0
        }
    }
}
