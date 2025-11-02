window.RAGMindApp = {
    init: function() {
        this.bindEvents();
        this.loadSystemStatus();
        this.loadChatHistory();
        console.log('RAG-Mind 应用已初始化');
    },

    bindEvents: function() {
        $('#toggleSidebar').click(this.toggleSidebar.bind(this));
        $('#newConversation').click(this.newConversation.bind(this));
        $('#sidebarOverlay').click(this.toggleSidebar.bind(this));
        $('#sendButton').click(this.sendMessage.bind(this));
        $('#messageInput').keypress(function(e) {
            if (e.which === 13) {
                window.RAGMindApp.sendMessage();
            }
        });
        $('#clearHistory').click(this.clearHistory.bind(this));
    },

    newConversation: function() {
        // 仅在本地开始新对话（不清除服务器端历史）
        const resetUI = () => {
            $('#chatMessages').empty();
            $('#messageInput').val('');
            $('#loadingIndicator').hide();
            $('#welcomeScreen').show();
            $('#chatMessages').hide();
        };

        resetUI();
        // 提示用户：服务器历史未被清除，可通过侧边栏的“清空”按钮手动清除
        alert('已开始新对话（服务器历史保留）。如需清空服务器历史，请使用侧边栏的“清空”按钮。');
    },

    toggleSidebar: function() {
        $('#sidebar').toggleClass('active');
        $('#sidebarOverlay').toggleClass('active');
    },

    sendMessage: function() {
        const message = $('#messageInput').val().trim();
        if (!message) return;

        $('#welcomeScreen').hide();
        $('#chatMessages').show();
        this.addMessage('user', message);
        $('#messageInput').val('');
        $('#loadingIndicator').show();
        this.sendQueryToAPI(message);
    },

    sendQueryToAPI: function(question) {
        $.ajax({
            url: '/api/query',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({question: question}),
            success: function(response) {
                $('#loadingIndicator').hide();
                if (response.success) {
                    // Pass chunk indices (if any) to support feedback mapping
                    const indices = response.data.chunk_indices || [];
                    window.RAGMindApp.addMessage('ai', response.data.answer, {chunk_indices: indices});
                    window.RAGMindApp.loadChatHistory();
                } else {
                    window.RAGMindApp.addMessage('ai', '抱歉，处理您的问题时出现错误：' + response.message);
                }
            },
            error: function() {
                $('#loadingIndicator').hide();
                window.RAGMindApp.addMessage('ai', '抱歉，网络连接出现问题，请稍后重试。');
            }
        });
    },

    addMessage: function(type, content, meta) {
        const headerHtml = type === 'user'
            ? `<i class="fas fa-user"></i> 您`
            : `<img src="/static/img/image.png" class="message-avatar" alt="RAG-Mind"/> RAG-Mind`;
        const messageElement = $(
            `<div class="chat-message ${type} fade-in">
                <div class="message-bubble ${type}">
                    <div class="message-header">
                        ${headerHtml}
                    </div>
                    <div class="message-content">${this.formatMessageContent(content)}</div>
                    <div class="message-timestamp">${this.getCurrentTime()}</div>
                </div>
            </div>`
        );

        // If AI message and meta contains chunk indices, append feedback buttons
        if (type === 'ai' && meta && Array.isArray(meta.chunk_indices) && meta.chunk_indices.length > 0) {
            const btnGroup = $(`
                <div class="mt-2 feedback-btns text-end">
                    <button class="btn btn-sm btn-outline-success me-1 feedback-up">有帮助</button>
                    <button class="btn btn-sm btn-outline-secondary feedback-down">无帮助</button>
                </div>
            `);

            // attach handler
            btnGroup.find('.feedback-up').click(function() {
                window.RAGMindApp.submitFeedback(meta.chunk_indices, 'up', $(this));
            });
            btnGroup.find('.feedback-down').click(function() {
                window.RAGMindApp.submitFeedback(meta.chunk_indices, 'down', $(this));
            });

            messageElement.find('.message-bubble').append(btnGroup);
        }

        $('#chatMessages').append(messageElement);
        this.scrollToBottom();
    },

    submitFeedback: function(chunk_indices, action, $button) {
        // Disable buttons immediately to avoid duplicate clicks
        const $btns = $button.closest('.feedback-btns').find('button');
        $btns.prop('disabled', true);

        $.ajax({
            url: '/api/feedback',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({chunk_indices: chunk_indices, action: action}),
            success: function(response) {
                if (response.success) {
                    $button.closest('.feedback-btns').html('<span class="text-success">已提交反馈，感谢！</span>');
                } else {
                    $btns.prop('disabled', false);
                    alert('提交反馈失败：' + (response.message || '未知错误'));
                }
            },
            error: function() {
                $btns.prop('disabled', false);
                alert('提交反馈失败，网络错误');
            }
        });
    },

    formatMessageContent: function(content) {
        return content.replace(/\n/g, '<br>');
    },

    summarizeText: function(content, length) {
        if (!content) return '';
        // 将换行折成空格，去掉多余空白
        const plain = content.replace(/\s+/g, ' ').trim();
        if (plain.length <= (length || 100)) return plain;
        return plain.substring(0, length || 100) + '...';
    },

    getCurrentTime: function() {
        const now = new Date();
        return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
    },

    scrollToBottom: function() {
        const chatMessages = $('#chatMessages');
        // 使用平滑滚动，稍作延迟以等待 DOM 渲染（例如图片或富文本）
        setTimeout(function() {
            try {
                // jQuery animate 平滑滚动（兼容性好）
                chatMessages.animate({ scrollTop: chatMessages[0].scrollHeight }, 300);
            } catch (e) {
                // 回退到直接设置
                chatMessages.scrollTop(chatMessages[0].scrollHeight);
            }
        }, 50);
    },

    loadSystemStatus: function() {
        $.get('/api/status', function(response) {
            if (response.success) {
                const stats = response.data;
                const statusHtml = `
                    <div class="row text-center">
                        <div class="col-6">
                            <i class="fas fa-database"></i>
                            <div>文本块: ${stats.total_chunks}</div>
                        </div>
                        <div class="col-6">
                            <i class="fas fa-file-alt"></i>
                            <div>文件: ${stats.sources.length}</div>
                        </div>
                    </div>
                `;
                $('#systemStatus').html(statusHtml);
            }
        });
    },

    loadChatHistory: function() {
        $.get('/api/history', function(response) {
            if (response.success) {
                const history = response.data;
                const historyList = $('#historyList');

                // 缓存历史到 app 对象，供点击历史时装载到聊天区使用
                window.RAGMindApp._history = history || [];

                if (history.length === 0) {
                    historyList.html(`
                        <div class="text-center text-muted mt-5">
                            <i class="fas fa-inbox fa-2x"></i>
                            <p class="mt-2">暂无历史记录</p>
                        </div>
                    `);
                    return;
                }

                let historyHtml = '';
                const recentHistory = history.slice(-10).reverse();

                recentHistory.forEach(function(item) {
                    const excerpt = window.RAGMindApp.summarizeText(item.answer, 100);
                    historyHtml += `
                        <div class="history-item" onclick="window.RAGMindApp.loadHistoryItem(${item.id})">
                            <div class="history-question">${item.question}</div>
                            <div class="history-answer">${excerpt}</div>
                            <div class="history-timestamp">${window.RAGMindApp.formatHistoryTime(item.timestamp)}</div>
                        </div>
                    `;
                });

                historyList.html(historyHtml);
                // 如果当前聊天窗口可见，则将历史重放到聊天区并滚到底
                if ($('#chatMessages').is(':visible')) {
                    // 清空并展示最近历史到聊天区（可选：只展示 ID 对应记录）
                    $('#chatMessages').empty();
                    const recent = history.slice(-10);
                    recent.forEach(function(item) {
                        window.RAGMindApp.addMessage('user', item.question);
                        window.RAGMindApp.addMessage('ai', item.answer);
                    });
                    window.RAGMindApp.scrollToBottom();
                }
            }
        });
    },

    formatHistoryTime: function(timestamp) {
        return new Date().toLocaleDateString();
    },

    loadHistoryItem: function(id) {
        // 从缓存中查找历史项并在聊天窗口中显示完整问与答
        const history = this._history || [];
        const item = history.find(h => h.id === id);

        if (!item) {
            // 若未缓存，尝试刷新历史后再查找（简单实现）
            this.loadChatHistory();
            alert('未找到该历史项，已刷新历史列表，请重试。');
            return;
        }

        // 切换 UI：隐藏欢迎屏，显示聊天区，清空当前消息
        $('#welcomeScreen').hide();
        $('#chatMessages').show();
        $('#chatMessages').empty();

        // 将历史的问题和回答按消息重放到聊天区域
        this.addMessage('user', item.question);
        this.addMessage('ai', item.answer);

        // 关闭侧边栏
        if ($('#sidebar').hasClass('active')) this.toggleSidebar();
    },

    clearHistory: function() {
        if (confirm('确定要清空所有聊天历史记录吗？')) {
            $.ajax({
                url: '/api/history/clear',
                method: 'DELETE',
                success: function(response) {
                    if (response.success) {
                        $('#historyList').html(`
                            <div class="text-center text-muted mt-5">
                                <i class="fas fa-inbox fa-2x"></i>
                                <p class="mt-2">暂无历史记录</p>
                            </div>
                        `);
                        alert('历史记录已清空');
                    }
                }
            });
        }
    }
};
