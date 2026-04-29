import { Request } from 'express'
import { StatusCodes } from 'http-status-codes'
import { InternalFlowiseError } from '../errors/internalFlowiseError'

type Pagination = {
    page: number
    limit: number
    searchTerm?: string
}

export const getPageLimitAndSearchParams = (req: Request): Pagination => {
    // by default assume no pagination
    let page = -1
    let limit = -1
    let searchTerm: string | undefined = undefined
    if (req.query.page) {
        // if page is provided, make sure it's a positive number
        page = parseInt(req.query.page as string)
        if (page < 0) {
            throw new InternalFlowiseError(StatusCodes.PRECONDITION_FAILED, `Error: page cannot be negative!`)
        }
    }
    if (req.query.limit) {
        // if limit is provided, make sure it's a positive number
        limit = parseInt(req.query.limit as string)
        if (limit < 0) {
            throw new InternalFlowiseError(StatusCodes.PRECONDITION_FAILED, `Error: limit cannot be negative!`)
        }
    }
    if (req.query.searchTerm) {
        searchTerm = req.query.searchTerm as string
    }
    return { page, limit, searchTerm }
}
