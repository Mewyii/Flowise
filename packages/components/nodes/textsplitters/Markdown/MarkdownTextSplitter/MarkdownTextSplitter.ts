import { MarkdownTextSplitter } from '@langchain/textsplitters'
import { INode, INodeData, INodeParams } from '../../../../src/Interface'
import { getBaseClasses } from '../../../../src/utils'
import { EnhancedMarkdownTextSplitter, EnhancedMarkdownTextSplitterParams } from '../EnhancedMarkdownTextSplitter'

class MarkdownTextSplitter_TextSplitters implements INode {
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
        this.label = 'Markdown Text Splitter'
        this.name = 'markdownTextSplitter'
        this.version = 1.1
        this.type = 'MarkdownTextSplitter'
        this.icon = 'markdownTextSplitter.svg'
        this.category = 'Text Splitters'
        this.description = `Split your content into documents based on the Markdown headers`
        this.baseClasses = [this.type, ...getBaseClasses(MarkdownTextSplitter)]
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
                label: 'Split by Headers',
                name: 'splitByHeaders',
                type: 'options',
                description: 'Split documents at specified header levels. Headers will be included with their content.',
                default: 'disabled',
                options: [
                    {
                        label: 'Disabled',
                        name: 'disabled'
                    },
                    {
                        label: '# Headers (H1)',
                        name: 'h1'
                    },
                    {
                        label: '## Headers (H2)',
                        name: 'h2'
                    },
                    {
                        label: '### Headers (H3)',
                        name: 'h3'
                    },
                    {
                        label: '#### Headers (H4)',
                        name: 'h4'
                    },
                    {
                        label: '##### Headers (H5)',
                        name: 'h5'
                    },
                    {
                        label: '###### Headers (H6)',
                        name: 'h6'
                    }
                ],
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
        const splitByHeaders = nodeData.inputs?.splitByHeaders as string
        const mergeSmallChunks = nodeData.inputs?.mergeSmallChunks as string

        const obj = {} as EnhancedMarkdownTextSplitterParams

        if (chunkSize) obj.chunkSize = parseInt(chunkSize, 10)
        if (chunkOverlap) obj.chunkOverlap = parseInt(chunkOverlap, 10)
        if (mergeSmallChunks) obj.mergeSmallChunksEnabled = mergeSmallChunks === 'enabled'
        if (splitByHeaders) obj.splitByHeaders = splitByHeaders

        const splitter = new EnhancedMarkdownTextSplitter(obj)

        return splitter
    }
}

module.exports = { nodeClass: MarkdownTextSplitter_TextSplitters }
