import { Document } from '@langchain/core/documents'
import { NodeHtmlMarkdown } from 'node-html-markdown'
import { INode, INodeData, INodeParams } from '../../../../src/Interface'
import { getBaseClasses } from '../../../../src/utils'
import { EnhancedMarkdownTextSplitter, EnhancedMarkdownTextSplitterParams } from '../EnhancedMarkdownTextSplitter'

class HtmlToMarkdownTextSplitter_TextSplitters implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    inputs: INodeParams[]

    constructor() {
        this.label = 'HtmlToMarkdown Text Splitter'
        this.name = 'htmlToMarkdownTextSplitter'
        this.version = 1.0
        this.type = 'HtmlToMarkdownTextSplitter'
        this.icon = 'htmlToMarkdownTextSplitter.svg'
        this.category = 'Text Splitters'
        this.description = `Converts Html to Markdown and then split your content into documents based on the Markdown headers`
        this.baseClasses = [this.type, ...getBaseClasses(HtmlToMarkdownTextSplitter)]
        this.inputs = [
            {
                label: 'Chunk Size',
                name: 'chunkSize',
                type: 'number',
                description: 'Number of characters in each chunk. Default is 1000.',
                default: 1000,
                optional: true
            },
            {
                label: 'Chunk Overlap',
                name: 'chunkOverlap',
                type: 'number',
                description: 'Number of characters to overlap between chunks. Default is 200.',
                default: 200,
                optional: true
            },
            {
                label: 'Merge Small Chunks',
                name: 'mergeSmallChunks',
                type: 'options',
                description:
                    'If enabled, merge adjacent chunks after converting HTML to Markdown until the chunk size limit would be exceeded.',
                default: 'enabled',
                options: [
                    {
                        label: 'Enabled',
                        name: 'enabled'
                    },
                    {
                        label: 'Disabled',
                        name: 'disabled'
                    }
                ],
                optional: true
            }
        ]
    }

    async init(nodeData: INodeData): Promise<any> {
        const chunkSize = nodeData.inputs?.chunkSize as string
        const chunkOverlap = nodeData.inputs?.chunkOverlap as string
        const mergeSmallChunks = nodeData.inputs?.mergeSmallChunks as string

        const obj = {} as EnhancedMarkdownTextSplitterParams

        if (chunkSize) obj.chunkSize = parseInt(chunkSize, 10)
        if (chunkOverlap) obj.chunkOverlap = parseInt(chunkOverlap, 10)
        if (mergeSmallChunks) obj.mergeSmallChunksEnabled = mergeSmallChunks === 'enabled'

        const splitter = new HtmlToMarkdownTextSplitter(obj)

        return splitter
    }
}
class HtmlToMarkdownTextSplitter extends EnhancedMarkdownTextSplitter implements EnhancedMarkdownTextSplitterParams {
    constructor(fields?: EnhancedMarkdownTextSplitterParams) {
        {
            super(fields)
        }
    }
    splitText(text: string): Promise<string[]> {
        const markdown = NodeHtmlMarkdown.translate(text)
        return super.splitText(markdown)
    }
    async splitDocuments(documents: Document[]): Promise<Document[]> {
        const results: Document[] = []

        for (const [index, doc] of documents.entries()) {
            const chunks = await this.splitText(doc.pageContent)
            const mergedChunks = this.mergeAdjacentChunks(chunks)

            results.push(...this.getChunksWithMetaData(mergedChunks, doc.metadata, index))
        }
        return results
    }
}
module.exports = { nodeClass: HtmlToMarkdownTextSplitter_TextSplitters }
