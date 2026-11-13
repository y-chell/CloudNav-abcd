import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import { AIConfig } from "../types";

export class AIRequestError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'AIRequestError';
    }
}

const getErrorMessage = (error: unknown): string =>
    error instanceof Error ? error.message : String(error);

/**
 * 把用户填写的 Base URL 归一化成 chat/completions 地址。
 * 支持 https://host、https://host/v1、https://host/v1/ 以及完整的 .../chat/completions。
 */
export const resolveChatCompletionsUrl = (rawBaseUrl: string): string => {
    const baseUrl = (rawBaseUrl || '').trim().replace(/\/+$/, '');
    if (!baseUrl) {
        throw new AIRequestError('请填写 Base URL（例如 https://api.openai.com/v1）');
    }

    let parsed: URL;
    try {
        parsed = new URL(baseUrl);
    } catch {
        throw new AIRequestError(`Base URL 格式不正确：${baseUrl}`);
    }

    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        throw new AIRequestError('Base URL 只支持 http/https');
    }

    if (/\/chat\/completions$/i.test(parsed.pathname)) {
        return parsed.toString();
    }

    if (!parsed.pathname || parsed.pathname === '/') {
        parsed.pathname = '/v1/chat/completions';
    } else {
        parsed.pathname = `${parsed.pathname.replace(/\/+$/, '')}/chat/completions`;
    }
    return parsed.toString();
};

const readApiError = async (response: Response): Promise<string> => {
    const body = await response.text();
    try {
        const data = JSON.parse(body);
        const message = data?.error?.message || data?.message;
        if (message) return String(message);
    } catch {
        // 不是 JSON，按原文返回。
    }
    return body.trim().slice(0, 500) || response.statusText;
};

/**
 * Helper to call OpenAI Compatible API
 */
const callOpenAICompatible = async (config: AIConfig, systemPrompt: string, userPrompt: string): Promise<string> => {
    if (!config.model || !config.model.trim()) {
        throw new AIRequestError('请填写模型名称');
    }

    const url = resolveChatCompletionsUrl(config.baseUrl);

    let response: Response;
    try {
        response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${config.apiKey}`
            },
            body: JSON.stringify({
                model: config.model.trim(),
                messages: [
                    { role: "system", content: systemPrompt },
                    { role: "user", content: userPrompt }
                ],
                // 不发送 temperature：GPT-5/o1/o3/o4 等推理模型只接受默认值 1，
                // 传 0.7 会直接返回 400。
                stream: false
            })
        });
    } catch (error) {
        throw new AIRequestError(
            `请求 ${url} 失败：${getErrorMessage(error)}。请检查 Base URL 和网络；如果是浏览器跨域限制，请改用允许 CORS 的 API 网关。`
        );
    }

    if (!response.ok) {
        throw new AIRequestError(`API 返回 HTTP ${response.status}：${await readApiError(response)}`);
    }

    const data = await response.json();
    const content = data?.choices?.[0]?.message?.content;
    if (typeof content === 'string' && content.trim()) {
        return content.trim();
    }
    if (Array.isArray(content)) {
        const text = content
            .map((part: any) => typeof part === 'string' ? part : part?.text || '')
            .join('')
            .trim();
        if (text) return text;
    }
    throw new AIRequestError('API 返回成功，但 choices[0].message.content 为空');
};

const cleanDescription = (text: string): string => {
    const cleaned = text
        .trim()
        .replace(/^["'“”‘’]+|["'“”‘’]+$/g, '')
        .trim();
    if (!cleaned) {
        throw new AIRequestError('AI 返回了空描述');
    }
    if (cleaned.length > 200) {
        throw new AIRequestError(`AI 返回内容过长（${cleaned.length} 字），可能不是简短描述`);
    }
    return cleaned;
};

const generateGeminiText = async (config: AIConfig, prompt: string): Promise<string> => {
    if (!config.model || !config.model.trim()) {
        throw new AIRequestError('请填写模型名称');
    }

    try {
        const ai = new GoogleGenAI({ apiKey: config.apiKey });
        const response: GenerateContentResponse = await ai.models.generateContent({
            model: config.model.trim(),
            contents: prompt,
        });
        const text = response.text?.trim();
        if (!text) throw new AIRequestError('Gemini 返回了空内容');
        return text;
    } catch (error) {
        if (error instanceof AIRequestError) throw error;
        throw new AIRequestError(`Gemini 调用失败：${getErrorMessage(error)}`);
    }
};

/**
 * Uses configured AI to generate a description
 */
export const generateLinkDescription = async (title: string, url: string, config: AIConfig): Promise<string> => {
    if (!config.apiKey || !config.apiKey.trim()) {
        throw new AIRequestError('请先在设置中配置 API Key');
    }

    const prompt = `
      Title: ${title}
      URL: ${url}
      Please write a very short description (max 15 words) in Chinese (Simplified) that explains what this website is for. Return ONLY the description text. No quotes.
  `;

    if (config.provider === 'gemini') {
        return cleanDescription(await generateGeminiText(config, `I have a website bookmark. ${prompt}`));
    }

    return cleanDescription(await callOpenAICompatible(
        config,
        "You are a helpful assistant that summarizes website bookmarks.",
        prompt
    ));
};

/**
 * Suggests a category
 */
export const suggestCategory = async (title: string, url: string, categories: {id: string, name: string}[], config: AIConfig): Promise<string | null> => {
    if (!config.apiKey || !config.apiKey.trim()) return null;

    const catList = categories.map(c => `${c.id}: ${c.name}`).join('\n');
    const prompt = `
        Website: "${title}" (${url})

        Available Categories:
        ${catList}

        Return ONLY the 'id' of the best matching category. If unsure, return 'common'.
    `;

    try {
        const raw = config.provider === 'gemini'
            ? await generateGeminiText(config, `Task: Categorize this website.\n${prompt}`)
            : await callOpenAICompatible(
                config,
                "You are an intelligent classification assistant. You only output the category ID.",
                prompt
            );

        const normalized = raw.trim().replace(/^["'`]+|["'`]+$/g, '').trim();
        const byId = categories.find(c => c.id === normalized);
        if (byId) return byId.id;
        const byName = categories.find(c => c.name === normalized);
        if (byName) return byName.id;

        console.warn('AI category suggestion returned an unknown category:', normalized);
        return null;
    } catch (error) {
        console.error('AI category suggestion failed:', getErrorMessage(error));
        return null;
    }
};
