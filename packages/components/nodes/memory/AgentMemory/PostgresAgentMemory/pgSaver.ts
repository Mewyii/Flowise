import { BaseMessage } from '@langchain/core/messages'
import { RunnableConfig } from '@langchain/core/runnables'
import { BaseCheckpointSaver, Checkpoint, CheckpointMetadata } from '@langchain/langgraph'
import { DataSource, QueryRunner } from 'typeorm'
import { IMessage, MemoryMethods } from '../../../../src/Interface'
import { mapChatMessageToBaseMessage } from '../../../../src/utils'
import { CheckpointTuple, SaverOptions, SerializerProtocol } from '../interface'

export class PostgresSaver extends BaseCheckpointSaver implements MemoryMethods {
    protected isSetup: boolean
    config: SaverOptions
    threadId: string
    tableName = 'checkpoints'

    constructor(config: SaverOptions, serde?: SerializerProtocol<Checkpoint>) {
        super(serde)
        this.config = config
        const { threadId } = config
        this.threadId = threadId
    }

    sanitizeTableName(tableName: string): string {
        // Trim and normalize case, turn whitespace into underscores
        tableName = tableName.trim().toLowerCase().replace(/\s+/g, '_')

        // Validate using a regex (alphanumeric and underscores only)
        if (!/^[a-zA-Z0-9_]+$/.test(tableName)) {
            throw new Error('Invalid table name')
        }

        return tableName
    }

    private async getDataSource(): Promise<DataSource> {
        const { datasourceOptions } = this.config
        if (!datasourceOptions) {
            throw new Error('No datasource options provided')
        }
        // Prevent using default MySQL port, otherwise will throw uncaught error and crashing the app
        if (datasourceOptions.port === 3306) {
            throw new Error('Invalid port number')
        }
        const dataSource = new DataSource(datasourceOptions)
        await dataSource.initialize()
        return dataSource
    }

    private async setup(dataSource: DataSource): Promise<void> {
        let queryRunner: QueryRunner | undefined

        if (this.isSetup) {
            return
        }

        try {
            queryRunner = dataSource.createQueryRunner()
            const tableName = this.sanitizeTableName(this.tableName)
            await queryRunner.manager.query(`
CREATE TABLE IF NOT EXISTS ${tableName} (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_id TEXT,
    checkpoint BYTEA,
    metadata BYTEA,
    PRIMARY KEY (thread_id, checkpoint_id));`)
        } catch (error) {
            console.error(`Error creating ${this.tableName} table`, error)
            throw new Error(`Error creating ${this.tableName} table`)
        } finally {
            if (queryRunner) await queryRunner.release()
        }

        this.isSetup = true
    }

    async getTuple(config: RunnableConfig): Promise<CheckpointTuple | undefined> {
        const dataSource = await this.getDataSource()
        let queryRunner: QueryRunner | undefined
        await this.setup(dataSource)

        const thread_id = config.configurable?.thread_id || this.threadId
        const checkpoint_id = config.configurable?.checkpoint_id
        const tableName = this.sanitizeTableName(this.tableName)

        if (checkpoint_id) {
            try {
                queryRunner = dataSource.createQueryRunner()
                const keys = [thread_id, checkpoint_id]
                const sql = `SELECT checkpoint, parent_id, metadata FROM ${tableName} WHERE thread_id = $1 AND checkpoint_id = $2`

                const rows = await queryRunner.manager.query(sql, keys)

                if (rows && rows.length > 0) {
                    return {
                        config,
                        checkpoint: (await this.serde.parse(rows[0].checkpoint.toString())) as Checkpoint,
                        metadata: (await this.serde.parse(rows[0].metadata.toString())) as CheckpointMetadata,
                        parentConfig: rows[0].parent_id
                            ? {
                                  configurable: {
                                      thread_id,
                                      checkpoint_id: rows[0].parent_id
                                  }
                              }
                            : undefined
                    }
                }
            } catch (error) {
                console.error(`Error retrieving ${tableName}`, error)
                throw new Error(`Error retrieving ${tableName}`)
            } finally {
                if (queryRunner) await queryRunner.release()
                await dataSource.destroy()
            }
        } else {
            try {
                queryRunner = dataSource.createQueryRunner()
                const keys = [thread_id]
                const sql = `SELECT thread_id, checkpoint_id, parent_id, checkpoint, metadata FROM ${tableName} WHERE thread_id = $1 ORDER BY checkpoint_id DESC LIMIT 1`

                const rows = await queryRunner.manager.query(sql, keys)

                if (rows && rows.length > 0) {
                    return {
                        config: {
                            configurable: {
                                thread_id: rows[0].thread_id,
                                checkpoint_id: rows[0].checkpoint_id
                            }
                        },
                        checkpoint: (await this.serde.parse(rows[0].checkpoint)) as Checkpoint,
                        metadata: (await this.serde.parse(rows[0].metadata)) as CheckpointMetadata,
                        parentConfig: rows[0].parent_id
                            ? {
                                  configurable: {
                                      thread_id: rows[0].thread_id,
                                      checkpoint_id: rows[0].parent_id
                                  }
                              }
                            : undefined
                    }
                }
            } catch (error) {
                console.error(`Error retrieving ${tableName}`, error)
                throw new Error(`Error retrieving ${tableName}`)
            } finally {
                if (queryRunner) await queryRunner.release()
                await dataSource.destroy()
            }
        }
        return undefined
    }

    async *list(config: RunnableConfig, limit?: number, before?: RunnableConfig): AsyncGenerator<CheckpointTuple> {
        const dataSource = await this.getDataSource()
        let queryRunner: QueryRunner | undefined

        await this.setup(dataSource)
        const thread_id = config.configurable?.thread_id || this.threadId
        const tableName = this.sanitizeTableName(this.tableName)
        let sql = `SELECT thread_id, checkpoint_id, parent_id, checkpoint, metadata FROM ${tableName} WHERE thread_id = $1`
        const args = [thread_id]

        if (before?.configurable?.checkpoint_id) {
            sql += ' AND checkpoint_id < $2'
            args.push(before.configurable.checkpoint_id)
        }

        sql += ' ORDER BY checkpoint_id DESC'
        if (limit) {
            sql += ` LIMIT ${limit}`
        }

        try {
            queryRunner = dataSource.createQueryRunner()
            const rows = await queryRunner.manager.query(sql, args)

            if (rows && rows.length > 0) {
                for (const row of rows) {
                    yield {
                        config: {
                            configurable: {
                                thread_id: row.thread_id,
                                checkpoint_id: row.checkpoint_id
                            }
                        },
                        checkpoint: (await this.serde.parse(rows[0].checkpoint.toString())) as Checkpoint,
                        metadata: (await this.serde.parse(rows[0].metadata.toString())) as CheckpointMetadata,
                        parentConfig: row.parent_id
                            ? {
                                  configurable: {
                                      thread_id: row.thread_id,
                                      checkpoint_id: row.parent_id
                                  }
                              }
                            : undefined
                    }
                }
            }
        } catch (error) {
            console.error(`Error listing ${tableName}`, error)
            throw new Error(`Error listing ${tableName}`)
        } finally {
            if (queryRunner) await queryRunner.release()
            await dataSource.destroy()
        }
    }

    async put(config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata): Promise<RunnableConfig> {
        const dataSource = await this.getDataSource()
        let queryRunner: QueryRunner | undefined
        await this.setup(dataSource)

        if (!config.configurable?.checkpoint_id) return {}
        try {
            queryRunner = dataSource.createQueryRunner()
            const row = [
                config.configurable?.thread_id || this.threadId,
                checkpoint.id,
                config.configurable?.checkpoint_id,
                Buffer.from(this.serde.stringify(checkpoint)), // Encode to binary
                Buffer.from(this.serde.stringify(metadata)) // Encode to binary
            ]
            const tableName = this.sanitizeTableName(this.tableName)

            const query = `INSERT INTO ${tableName} (thread_id, checkpoint_id, parent_id, checkpoint, metadata)
                           VALUES ($1, $2, $3, $4, $5)
                           ON CONFLICT (thread_id, checkpoint_id)
                           DO UPDATE SET checkpoint = EXCLUDED.checkpoint, metadata = EXCLUDED.metadata`

            await queryRunner.manager.query(query, row)
        } catch (error) {
            console.error('Error saving checkpoint', error)
            throw new Error('Error saving checkpoint')
        } finally {
            if (queryRunner) await queryRunner.release()
            await dataSource.destroy()
        }

        return {
            configurable: {
                thread_id: config.configurable?.thread_id || this.threadId,
                checkpoint_id: checkpoint.id
            }
        }
    }

    async delete(threadId: string): Promise<void> {
        if (!threadId) {
            return
        }

        const dataSource = await this.getDataSource()
        let queryRunner: QueryRunner | undefined
        await this.setup(dataSource)
        const tableName = this.sanitizeTableName(this.tableName)

        const query = `DELETE FROM "${tableName}" WHERE thread_id = $1;`

        try {
            queryRunner = dataSource.createQueryRunner()
            await queryRunner.manager.query(query, [threadId])
        } catch (error) {
            console.error(`Error deleting thread_id ${threadId}`, error)
        } finally {
            if (queryRunner) await queryRunner.release()
            await dataSource.destroy()
        }
    }

    async getChatMessages(
        overrideSessionId = '',
        returnBaseMessages = false,
        prependMessages?: IMessage[]
    ): Promise<IMessage[] | BaseMessage[]> {
        if (!overrideSessionId) return []

        const chatMessage = await this.config.appDataSource.getRepository(this.config.databaseEntities['ChatMessage']).find({
            where: {
                sessionId: overrideSessionId,
                chatflowid: this.config.chatflowid
            },
            order: {
                createdDate: 'ASC'
            }
        })

        if (prependMessages?.length) {
            chatMessage.unshift(...prependMessages)
        }

        if (returnBaseMessages) {
            return await mapChatMessageToBaseMessage(chatMessage, this.config.orgId)
        }

        let returnIMessages: IMessage[] = []
        for (const m of chatMessage) {
            returnIMessages.push({
                message: m.content as string,
                type: m.role
            })
        }
        return returnIMessages
    }

    async addChatMessages(): Promise<void> {
        // Empty as it's not being used
    }

    async clearChatMessages(overrideSessionId = ''): Promise<void> {
        await this.delete(overrideSessionId)
    }
}
